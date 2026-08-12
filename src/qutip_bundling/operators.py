"""Pure operator transforms for the stochastic bundling method.

These functions implement the *method itself*, factored away from any
solver. They are the building blocks of

    S. Adhikari and R. Baer, "Stochastically Bundled Dissipators for the
    Quantum Master Equation", J. Chem. Theory Comput. 2025, 21, 4142-4150.
    https://doi.org/10.1021/acs.jctc.5c00145

There are three independent transforms. They take operator lists as input
and return operator lists (or a single operator) as output -- no
Hamiltonians, no initial states, no time grids. The user composes them
with their own ``qutip.mesolve`` call (or ``mcsolve``, ``smesolve``, or
a custom propagator).

build_collapse_ops
    Take bare Lindblad operators ``L_alpha`` and the spectral function
    ``gamma(omega)``, return the collapse operators
    ``c_alpha = sqrt(gamma(omega_alpha)) * L_alpha`` that QuTiP expects.

bundle
    Take any collapse-operator list and return ``M`` randomly bundled
    operators. The bundled dissipator is an unbiased estimator of the
    full one and stays in Lindblad form.

lamb_shift_hamiltonian
    Take the bare ``L_alpha`` and the *imaginary* part of the spectral
    function, return the Davies Lamb-shift Hamiltonian
    ``H_LS = sum_alpha imag_gamma(omega_alpha) * L_alpha^dag L_alpha``.
    Built once at setup -- the user adds it to their system Hamiltonian.

The Lamb shift is computed from the *original* operators, not the
bundled ones. It is part of the renormalized Hamiltonian (eq. 4 of the
paper), so it has a deterministic closed form; bundling is a Monte Carlo
trick for the dissipator only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import qutip

from ._spectral import SpectralInput, evaluate_spectral

__all__ = [
    "davies_operators",
    "build_collapse_ops",
    "bundle",
    "bundle_from_phases",
    "lamb_shift_hamiltonian",
    "prepare_bundled_dynamics",
    "random_phases",
    "BundledOps",
]


# Distributions for the random coefficients r_{m,alpha}.
# All satisfy  E[r] = 0  and  E[r conj(r)] = 1  (eq. 8 of the paper).
_DISTRIBUTIONS = ("phase", "pm1", "uniform")
# Safety factor for the backward-error floor on projector blocks. Dense basis
# transformations accumulate O(eps * N * ||X||) roundoff; the Frobenius norm
# supplies a cheap conservative scale bound.
_COUPLING_ROUNDOFF_FACTOR = 512.0


# --------------------------------------------------------------------------
# Random coefficients
# --------------------------------------------------------------------------
def random_phases(
    n_bundles: int,
    n_ops: int,
    distribution: str = "phase",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Draw a random coefficient matrix ``r`` of shape ``(M, N_L)``.

    The distributions are all zero-mean, unit second-moment:

    * ``"phase"``   -- ``exp(i*theta)`` on the unit circle (paper default);
    * ``"pm1"``     -- discrete ``{-1, +1}``;
    * ``"uniform"`` -- real uniform on ``[-sqrt(3), sqrt(3)]``.
    """
    if distribution not in _DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {_DISTRIBUTIONS}, got {distribution!r}"
        )
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    shape = (n_bundles, n_ops)
    if distribution == "phase":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=shape)
        return np.exp(1j * theta)
    if distribution == "pm1":
        return rng.choice(np.array([-1.0, 1.0]), size=shape).astype(complex)
    s3 = math.sqrt(3.0)
    return rng.uniform(-s3, s3, size=shape).astype(complex)


# --------------------------------------------------------------------------
# 0) davies_operators -- build the collapse operators from scratch, the
#    strict secular way, with degeneracies and the sign convention handled.
# --------------------------------------------------------------------------
def _cluster_sorted(values: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Cluster sorted real values without allowing tolerance-chain bridging."""
    if len(values) == 0:
        return []
    groups: list[list[int]] = [[0]]
    first = float(values[0])
    for index in range(1, len(values)):
        value = float(values[index])
        if value - first <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
            first = value
    return [np.asarray(group, dtype=int) for group in groups]


def _coupling_roundoff_floor(X_eig: np.ndarray) -> float:
    """Smallest coupling block distinguishable from basis-change roundoff.

    The expression is homogeneous in ``X``: rescaling the physical coupling
    rescales the floor by the same amount, unlike a fixed absolute cutoff.
    """
    dimension = int(X_eig.shape[0])
    scale = float(np.linalg.norm(X_eig, ord="fro"))
    return _COUPLING_ROUNDOFF_FACTOR * np.finfo(float).eps * dimension * scale


def davies_operators(
    H: qutip.Qobj,
    X: qutip.Qobj,
    gamma: SpectralInput,
    *,
    imag_gamma: SpectralInput | float | None = None,
    threshold: float = 0.0,
    coupling_threshold: float = 0.0,
    lamb_shift_threshold: float | None = None,
    degeneracy_tol: float = 1e-5,
    return_bare: bool = False,
):
    """Build Davies/Bohr collapse operators directly from H and a coupling op.

    This is the recommended entry point. It diagonalizes the Hamiltonian,
    constructs projectors onto its (possibly degenerate) energy eigenspaces,
    groups all transitions with the same Bohr frequency, and applies the
    spectral weighting. The grouped construction is invariant under arbitrary
    rotations inside a degenerate eigenspace and retains the cross terms
    required by the strict secular Davies generator.

    The construction (Davies weak-coupling / eq. 6 of the paper):

        H = sum_e e * Pi_e
        A(omega) = sum_(e' - e = omega) Pi_e X Pi_e'
        c(omega) = sqrt(gamma(omega)) * A(omega)

    Thus a transition from the higher eigenspace ``e'`` to the lower
    eigenspace ``e`` has ``omega = e' - e > 0`` and, for the detailed-balance
    convention used by this package, the larger rate. Grouping is essential:
    replacing ``D[A(omega)]`` by a sum of dissipators for individual matrix
    elements drops interference terms and becomes basis-dependent whenever
    energies or Bohr frequencies are degenerate.

    Parameters
    ----------
    H : qutip.Qobj
        System Hamiltonian (Hermitian). Includes everything coherent
        *except* the Lamb shift, which is returned separately if requested.
    X : qutip.Qobj
        System operator through which the bath couples (e.g. a position
        or dipole operator).
    gamma : callable or array
        Real spectral function ``gamma(omega) >= 0``. If an array, it must
        be evaluated by the caller on the Bohr frequencies -- but since the
        Bohr frequencies are computed *inside* this function, a callable is
        strongly preferred here.
    imag_gamma : callable, float, or None
        If given, also return the Lamb-shift Hamiltonian
        ``sum imag_gamma(omega) * A(omega)^dag A(omega)`` built from the bare
        operators. ``None`` (default) skips it.
    threshold : float
        Drop a grouped operator when
        ``sqrt(gamma(omega)) * ||A(omega)||_F`` is below this. Default
        ``0.0`` keeps every nonzero frequency sector.
    coupling_threshold : float
        Prune an energy-eigenspace block ``Pi_e X Pi_e'`` when its Frobenius
        norm is below this, before equal-frequency blocks are summed. The
        comparison is basis-invariant inside degenerate eigenspaces.
        ``0.0`` (default) keeps every numerically resolvable block: an
        automatic, scale-covariant backward-error floor removes only
        basis-transformation roundoff. A positive value applies the larger
        of the requested physical cutoff and that numerical floor.
    lamb_shift_threshold : float or None
        Threshold passed to :func:`lamb_shift_hamiltonian` for dropping
        Lamb-shift terms. ``None`` (default) reuses ``threshold`` -- the
        original behavior -- but note the Lamb-shift filter compares against
        ``|imag_gamma|``, a third distinct quantity, so an aggressive
        operator ``threshold`` would otherwise silently prune Lamb-shift
        terms by an unrelated criterion. Set this explicitly (e.g. ``0.0``)
        to decouple the two.
    degeneracy_tol : float
        Absolute tolerance, in the energy units of ``H``, for identifying
        degenerate eigenvalues and equal Bohr frequencies. The default
        ``1e-10`` is intended to absorb diagonalization roundoff, not to
        perform a partial-secular approximation for physically distinct
        near-degenerate transitions. Set it smaller if the model contains
        meaningful splittings below this scale.
    return_bare : bool
        If True, also return the bare operators and Bohr frequencies, so
        you can pass them to :func:`lamb_shift_hamiltonian` yourself or
        inspect them.

    Returns
    -------
    c_ops : list of qutip.Qobj
        The gamma-weighted collapse operators, ready for ``mesolve`` or
        :func:`bundle`.
    H_LS : qutip.Qobj
        Lamb-shift Hamiltonian, only if ``imag_gamma`` is not None.
    (L_ops, omegas) : tuple
        Bare operators and Bohr frequencies, only if ``return_bare``.

    The return is a single value (``c_ops``) when neither ``imag_gamma``
    nor ``return_bare`` is set; otherwise a tuple in the order
    ``(c_ops[, H_LS][, (L_ops, omegas)])``.
    """
    if not callable(gamma):
        raise TypeError(
            "davies_operators needs gamma as a callable f(omega); the Bohr "
            "frequencies are computed internally so an array cannot be "
            "aligned. Use build_collapse_ops if you already have arrays."
        )
    if degeneracy_tol < 0:
        raise ValueError("degeneracy_tol must be non-negative.")

    H_arr = np.asarray(H.full())
    H_arr = 0.5 * (H_arr + H_arr.conj().T)
    if coupling_threshold < 0:
        raise ValueError("coupling_threshold must be non-negative.")
    evals, evecs = np.linalg.eigh(H_arr)
    X_eig = evecs.conj().T @ np.asarray(X.full()) @ evecs

    # 1) Build numerical energy eigenspaces. Projector blocks, unlike
    #    individual eigenvectors, are invariant under rotations inside an
    #    exactly degenerate subspace.
    energy_groups = _cluster_sorted(evals, degeneracy_tol)
    energies = np.asarray(
        [float(np.mean(evals[group])) for group in energy_groups],
        dtype=float,
    )

    # 2) Keep significant projector blocks Pi_e X Pi_e'. In the energy
    #    representation each block is just a submatrix of X_eig. Its
    #    Frobenius norm is invariant under basis rotations within either
    #    eigenspace and reduces to |<a|X|b>| for one-dimensional spaces.
    # A fixed 1e-14 floor is not reproducible: LAPACK implementations put
    # symmetry-forbidden zeros on different sides of that number. Use a
    # scale-covariant backward-error bound instead, so those null blocks
    # cannot become thousands of machine-dependent Davies sectors.
    numerical_floor = _coupling_roundoff_floor(X_eig)
    block_floor = max(float(coupling_threshold), numerical_floor)
    transitions: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for a_group, a_indices in enumerate(energy_groups):
        for b_group, b_indices in enumerate(energy_groups):
            block = X_eig[np.ix_(a_indices, b_indices)]
            if np.linalg.norm(block, ord="fro") <= block_floor:
                continue
            omega = float(energies[b_group] - energies[a_group])
            transitions.append((omega, a_indices, b_indices, block))

    # 3) Group all projector blocks with the same Bohr frequency. Requiring
    #    the full sector span to fit within the tolerance avoids accidental
    #    "chaining" of several genuinely distinct nearby frequencies.
    transitions.sort(key=lambda item: item[0])
    sectors: list[list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]] = []
    for transition in transitions:
        if (
            not sectors
            or transition[0] - sectors[-1][0][0] > degeneracy_tol
        ):
            sectors.append([transition])
        else:
            sectors[-1].append(transition)

    omega_all: list[float] = []
    bare_norms: list[float] = []
    for sector in sectors:
        frequencies = [transition[0] for transition in sector]
        omega = float(np.mean(frequencies))
        if abs(omega) <= degeneracy_tol:
            omega = 0.0
        omega_all.append(omega)
        # The projector blocks occupy orthogonal row/column subspaces, so
        # their squared Frobenius norms add. Avoid retaining one dense N x N
        # array per sector; that would double peak memory at large dimension.
        bare_norms.append(math.sqrt(sum(
            float(np.linalg.norm(transition[3], ord="fro")) ** 2
            for transition in sector
        )))

    omega_all_arr = np.asarray(omega_all, dtype=float)

    # 4) Evaluate one bath rate per frequency sector and apply the master
    #    threshold to the complete grouped operator.
    gammas = evaluate_spectral(gamma, omega_all_arr, name="gamma")
    if np.any(gammas < 0):
        raise ValueError("gamma must be non-negative.")
    sg_all = np.sqrt(gammas)
    weights = sg_all * np.asarray(bare_norms, dtype=float)
    keep = (weights >= threshold) & (weights != 0.0)

    L_ops: list[qutip.Qobj] = []
    omegas: list[float] = []
    c_ops: list[qutip.Qobj] = []
    for sector, omega, sg, k in zip(sectors, omega_all_arr, sg_all, keep):
        if not k:
            continue
        A_eig = np.zeros_like(X_eig)
        for _, a_indices, b_indices, block in sector:
            A_eig[np.ix_(a_indices, b_indices)] += block
        A = evecs @ A_eig @ evecs.conj().T
        bare = qutip.Qobj(A, dims=H.dims)
        L_ops.append(bare)
        omegas.append(float(omega))
        c_ops.append(float(sg) * bare)

    omegas_arr = np.asarray(omegas, dtype=float)
    ls_threshold = threshold if lamb_shift_threshold is None else lamb_shift_threshold
    out: list = [c_ops]
    if imag_gamma is not None:
        if L_ops:
            out.append(lamb_shift_hamiltonian(
                L_ops, omegas_arr, imag_gamma, threshold=ls_threshold
            ))
        else:
            out.append(0 * H)
    if return_bare:
        out.append((L_ops, omegas_arr))

    return out[0] if len(out) == 1 else tuple(out)


# --------------------------------------------------------------------------
# 1) build_collapse_ops
# --------------------------------------------------------------------------
def build_collapse_ops(
    L_ops: Sequence[qutip.Qobj],
    omegas: Sequence[float] | np.ndarray,
    gamma: SpectralInput,
    threshold: float = 0.0,
) -> list[qutip.Qobj]:
    """Build standard collapse operators from bare Lindblad operators.

    Implements ``c_alpha = sqrt(gamma(omega_alpha)) * L_alpha`` from the
    Davies formalism. ``gamma`` may be a callable ``f(omega) -> float`` or
    an array of the same length as ``L_ops`` (aligned with ``omegas``).

    .. note::
       This function trusts the ``omegas`` you pass and does not enforce a
       sign convention. The convention must be consistent with how you
       built ``L_alpha``: for a transition operator ``L = |a><b|`` the
       associated Bohr frequency should be ``omega = E_b - E_a`` so that a
       downward (energy-releasing) transition has ``omega > 0`` and -- for
       a detailed-balance ``gamma`` -- the larger rate. Getting this sign
       backwards makes the system anti-relax (heat up instead of cool).
       If you have ``H`` and a coupling operator rather than pre-built
       ``L_alpha``, prefer :func:`davies_operators`, which bakes in the
       correct sign.

    Operators whose weight ``sqrt(gamma(omega_alpha)) * ||L_alpha||`` falls
    below ``threshold`` are dropped -- the analogue of the ``sparcity``
    cutoff in the C++ reference implementation, and the same quantity
    :func:`davies_operators` thresholds on (for a Davies operator
    ``L = amp*|a><b|`` the Frobenius norm ``||L||_F`` equals ``|amp|``, so a given
    ``threshold`` selects the same operators through either entry point).
    Default ``threshold=0.0`` keeps everything.

    Parameters
    ----------
    L_ops
        Bare Lindblad operators ``L_alpha`` (e.g. the Bohr-projector form
        of eq. 6 in the paper). Length ``N_L``.
    omegas
        Bohr frequencies aligned with ``L_ops``. Length ``N_L``.
    gamma
        Real spectral function ``gamma(omega) >= 0``, callable or array.
    threshold
        Drop operators whose weight ``sqrt(gamma(omega_alpha)) * ||L_alpha||``
        is below this. Matches the semantics of ``davies_operators``.

    Returns
    -------
    list of qutip.Qobj
        Collapse operators ready for ``qutip.mesolve(..., c_ops=...)``.
    """
    L_ops = list(L_ops)
    omegas = np.asarray(omegas, dtype=float)
    if len(L_ops) != omegas.size:
        raise ValueError(
            f"len(L_ops)={len(L_ops)} != len(omegas)={omegas.size}"
        )
    gammas = evaluate_spectral(gamma, omegas, name="gamma")
    if np.any(gammas < 0):
        raise ValueError("gamma must be non-negative on every Bohr frequency.")

    sqrtg = np.sqrt(gammas)
    out: list[qutip.Qobj] = []
    for alpha, sg in enumerate(sqrtg):
        # Weight is the norm of the resulting collapse operator,
        # sqrt(gamma) * ||L||. This makes `threshold` mean the same thing
        # here as in davies_operators -- rate times coupling strength, the
        # single sparsity scalar of the StochLind C++ reference -- rather
        # than the rate alone. For a Davies operator L = amp*|a><b| the
        # Frobenius norm ||L||_F equals |amp|, so the entry points agree.
        weight = float(sg) * L_ops[alpha].norm("fro")
        if weight < threshold or weight == 0.0:
            continue
        out.append(sg * L_ops[alpha])
    return out


# --------------------------------------------------------------------------
# 2) bundle
# --------------------------------------------------------------------------
def bundle_from_phases(
    c_ops: Sequence[qutip.Qobj],
    phases: np.ndarray,
    max_memory_bytes: float = 256 * 1024 * 1024,
) -> list[qutip.Qobj]:
    """Build bundled operators from an explicit phase matrix.

    ``R_m = (1 / sqrt(M)) * sum_alpha phases[m, alpha] * c_ops[alpha]``.

    Low-level routine -- :func:`bundle` is the convenient front end. This
    is exposed because the jackknife estimator needs to reuse the *same*
    random draws for sub-bundles.

    Memory Guard / Streaming:
    If assembling all N_L operators into a single (N_L, N^2) array exceeds
    ``max_memory_bytes`` (default 256 MB), the stack is assembled and multiplied
    in streaming chunks of size ``chunk_size``. This reduces peak memory usage
    from O(N_L * N^2) to O(chunk_size * N^2) while retaining BLAS acceleration.
    """
    c_ops = list(c_ops)
    if len(c_ops) == 0:
        raise ValueError("c_ops is empty -- nothing to bundle.")
    phases = np.asarray(phases)
    if phases.ndim != 2 or phases.shape[1] != len(c_ops):
        raise ValueError(
            f"phases must have shape (M, {len(c_ops)}), got {phases.shape}."
        )

    M = phases.shape[0]
    N_L = len(c_ops)
    inv_sqrt_m = 1.0 / math.sqrt(M)

    ref0 = c_ops[0]
    dims, shape = ref0.dims, ref0.shape
    n_prod = shape[0] * shape[1]
    bytes_per_op = n_prod * 16  # 16 bytes per complex128
    chunk_size = max(1, int(max_memory_bytes // bytes_per_op)) if bytes_per_op > 0 else N_L

    scaled_phases = inv_sqrt_m * phases.astype(complex)

    if chunk_size >= N_L:
        # Full vectorized assembly in a single BLAS gemm
        stack = np.stack([np.asarray(op.full(), dtype=complex).ravel() for op in c_ops]) # (N_L, N*N)
        combined = scaled_phases @ stack # (M, N*N)
    else:
        # Streaming chunked assembly to bound peak memory to max_memory_bytes
        combined = np.zeros((M, n_prod), dtype=complex)
        for start in range(0, N_L, chunk_size):
            end = min(start + chunk_size, N_L)
            chunk_stack = np.stack([np.asarray(op.full(), dtype=complex).ravel()
                                    for op in c_ops[start:end]])
            combined += scaled_phases[:, start:end] @ chunk_stack

    return [qutip.Qobj(combined[m].reshape(shape), dims=dims) for m in range(M)]


def bundle(
    c_ops: Sequence[qutip.Qobj],
    M: int,
    distribution: str = "phase",
    rng: np.random.Generator | int | None = None,
) -> list[qutip.Qobj]:
    """Stochastically bundle ``c_ops`` into ``M`` operators (eq. 11).

    ``R_m = (1/sqrt(M)) * sum_alpha r_{m,alpha} * c_ops[alpha]``

    with ``r_{m,alpha}`` drawn from a zero-mean, unit-variance ensemble.
    The bundled dissipator
    ``D[rho] = sum_m ( R_m rho R_m^dag - 0.5 {R_m^dag R_m, rho} )`` is an
    unbiased estimator of the full dissipator for any ``M``, and still
    has Lindblad form, so dynamics remain CPTP.

    Parameters
    ----------
    c_ops
        Full collapse-operator list. Each entry should already include
        any ``sqrt(gamma)`` factor (use :func:`build_collapse_ops` if you
        have bare ``L_alpha`` plus a spectral function).
    M
        Number of bundled operators to produce. Must be >= 1.
    distribution
        ``"phase"``, ``"pm1"``, or ``"uniform"``; see :func:`random_phases`.
    rng
        Generator or seed for reproducibility.

    Returns
    -------
    list of qutip.Qobj
        ``M`` bundled collapse operators, ready for ``qutip.mesolve``.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}.")
    phases = random_phases(M, len(c_ops), distribution=distribution, rng=rng)
    return bundle_from_phases(c_ops, phases)


# --------------------------------------------------------------------------
# 3) lamb_shift_hamiltonian
# --------------------------------------------------------------------------
def lamb_shift_hamiltonian(
    L_ops: Sequence[qutip.Qobj],
    omegas: Sequence[float] | np.ndarray,
    imag_gamma: SpectralInput,
    threshold: float = 0.0,
) -> qutip.Qobj:
    """Build the Davies Lamb-shift Hamiltonian (eq. 4 of the paper).

    ``H_LS = sum_alpha imag_gamma(omega_alpha) * L_alpha^dag L_alpha``

    Computed from the *bare* Lindblad operators -- bundling never enters
    here. The result is a single Hermitian ``Qobj`` that the user adds to
    their system Hamiltonian once at setup; it does not change between
    time steps or bundled realizations.

    If ``imag_gamma`` is identically zero (the paper's test cases), the
    returned operator is exactly zero. Including the term has negligible
    cost relative to the dissipator regardless.

    Parameters
    ----------
    L_ops
        Bare Lindblad operators ``L_alpha`` (NOT bundled, NOT pre-multiplied
        by ``sqrt(gamma)``).
    omegas
        Bohr frequencies aligned with ``L_ops``.
    imag_gamma
        Imaginary part of the spectral function -- callable ``f(omega)``
        or array aligned with ``omegas``. Use ``imag_gamma=0`` (or a
        callable returning 0) to disable the Lamb shift.
    threshold
        Skip terms with ``|imag_gamma(omega_alpha)| < threshold``.

    Returns
    -------
    qutip.Qobj
        Hermitian Lamb-shift Hamiltonian.
    """
    L_ops = list(L_ops)
    omegas = np.asarray(omegas, dtype=float)
    if len(L_ops) != omegas.size:
        raise ValueError(
            f"len(L_ops)={len(L_ops)} != len(omegas)={omegas.size}"
        )

    # convenience: scalar 0 (or callable that returns 0) is common
    if not callable(imag_gamma) and np.isscalar(imag_gamma):
        if float(imag_gamma) == 0.0:
            return 0 * (L_ops[0].dag() * L_ops[0])
        imag_gamma_vals = np.full(omegas.shape, float(imag_gamma))
    else:
        imag_gamma_vals = evaluate_spectral(
            imag_gamma, omegas, name="imag_gamma"
        )

    H_LS = None
    for alpha, ig in enumerate(imag_gamma_vals):
        if abs(ig) < threshold or ig == 0.0:
            continue
        L = L_ops[alpha]
        contrib = float(ig) * (L.dag() * L)
        H_LS = contrib if H_LS is None else H_LS + contrib

    if H_LS is None:
        return 0 * (L_ops[0].dag() * L_ops[0])

    # symmetrize against floating-point drift; H_LS must be Hermitian
    return 0.5 * (H_LS + H_LS.dag())


# --------------------------------------------------------------------------
# Convenience composition of the three transforms
# --------------------------------------------------------------------------
@dataclass
class BundledOps:
    """Output of :func:`prepare_bundled_dynamics`.

    The user combines these with their own Hamiltonian and solver call:

        H_total = H_sys + bundled.H_lamb_shift
        result  = qutip.mesolve(H_total, rho0, tlist,
                                c_ops=bundled.c_ops, e_ops=...)
    """
    c_ops: list[qutip.Qobj]      #: M bundled collapse operators
    H_lamb_shift: qutip.Qobj      #: Lamb-shift Hamiltonian (Qobj, possibly zero)
    M: int                        #: bundle size
    extras: dict = field(default_factory=dict)


def prepare_bundled_dynamics(
    L_ops: Sequence[qutip.Qobj],
    omegas: Sequence[float] | np.ndarray,
    gamma: SpectralInput,
    M: int,
    imag_gamma: SpectralInput | float | None = None,
    *,
    distribution: str = "phase",
    rng: np.random.Generator | int | None = None,
    threshold: float = 0.0,
) -> BundledOps:
    """One-shot helper: full collapse list -> ``M`` bundled ops plus H_LS.

    Composes :func:`build_collapse_ops`, :func:`bundle` and
    :func:`lamb_shift_hamiltonian` in the natural order. Set
    ``imag_gamma=None`` (default) or ``0`` to skip the Lamb shift.

    Returns
    -------
    BundledOps
        ``.c_ops`` (length ``M``) and ``.H_lamb_shift`` (a Qobj). The user
        adds ``H_lamb_shift`` to their system Hamiltonian and passes
        ``c_ops`` to any QuTiP solver.
    """
    c_full = build_collapse_ops(L_ops, omegas, gamma, threshold=threshold)
    if not c_full:
        raise ValueError(
            "No collapse operators survived the threshold -- check gamma "
            "and threshold."
        )
    R = bundle(c_full, M, distribution=distribution, rng=rng)

    if imag_gamma is None:
        H_LS = 0 * (L_ops[0].dag() * L_ops[0])
    else:
        H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma,
                                       threshold=threshold)

    return BundledOps(c_ops=R, H_lamb_shift=H_LS, M=M,
                       extras={"n_full_ops": len(c_full)})
