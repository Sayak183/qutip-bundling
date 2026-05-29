"""Optional convenience wrappers: ensemble averaging and jackknife bias correction.

These are thin layers over :mod:`qutip_bundling.operators` plus a
master-equation solver. The pure operator transforms are the core API;
these helpers exist so you don't have to write the for-loop yourself.

The ``backend`` parameter selects the propagator:

* ``"qutip"`` (default) -- uses ``qutip.mesolve``. Convenient and fully
  featured but builds the Liouvillian superoperator, so memory scales
  as ``O(N_c * N^4)``. Fine for small systems (Hilbert dim up to ~30 on
  a typical laptop).

* ``"native"`` -- uses :func:`rk4_mesolve`, a classical RK4 stepper that
  operates directly on the density matrix. Memory scales as
  ``O((N_c + 1) * N^2)``, lifting the size ceiling dramatically. Useful
  precisely in the regime where bundling pays off, i.e. many collapse
  operators and/or large Hilbert space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import qutip

from .native_solver import rk4_mesolve
from .operators import bundle, bundle_from_phases, random_phases

__all__ = ["mesolve_ensemble", "mesolve_jackknife", "BundledResult"]


@dataclass
class BundledResult:
    """Result of an averaged bundled solve.

    Attributes
    ----------
    times
        Time grid, shape ``(n_times,)``.
    expect
        One real array per ``e_op``: the **mean** expectation value over all
        realizations. ``expect[k]`` has shape ``(n_times,)``.
    sem
        One real array per ``e_op``: the **standard error of the mean** across
        realizations. This is ``std / sqrt(n_realizations)`` and shrinks as you
        add realizations; use it for error bars on the estimated mean.
    std
        One real array per ``e_op``: the **standard deviation** of the
        observable across realizations (sample std, ``ddof=1``). This
        characterizes how noisy a *single* bundled trajectory is and does NOT
        shrink with more realizations. Computed as
        ``samples.std(axis=0, ddof=1)``.
    samples
        Raw per-realization data, shape ``(n_realizations, n_e_ops, n_times)``.
        Everything else (mean, sem, std, percentiles, custom fluctuation
        definitions, histograms) can be recomputed from this, so no
        information is lost to the summary statistics above.
    M
        Bundle size used.
    n_realizations
        Number of independent bundled solves that were averaged.
    extra
        Method-specific extras (e.g. the uncorrected mean for jackknife).
    """
    times: np.ndarray
    expect: list[np.ndarray]
    sem: list[np.ndarray]
    std: list[np.ndarray]
    samples: np.ndarray
    M: int
    n_realizations: int
    extra: dict = field(default_factory=dict)


def _solve(H, rho0, tlist, c_ops, e_ops, options, backend, substeps):
    """Backend dispatch. Returns expectation values as (n_e_ops, n_times) real."""
    if backend == "qutip":
        res = qutip.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops,
                              options=options)
        return np.real(np.asarray(res.expect))
    if backend == "native":
        res = rk4_mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops,
                           substeps=substeps)
        return np.asarray(res.expect)
    raise ValueError(f"backend must be 'qutip' or 'native', got {backend!r}")


def mesolve_ensemble(
    H,
    rho0,
    tlist,
    c_ops_full: Sequence[qutip.Qobj],
    M: int,
    e_ops: Sequence[qutip.Qobj],
    n_realizations: int = 16,
    distribution: str = "phase",
    rng: np.random.Generator | int | None = None,
    options=None,
    *,
    backend: str = "qutip",
    substeps: int = 1,
) -> BundledResult:
    """Average ``n_realizations`` independent bundled mesolves.

    Each realization draws a fresh phase matrix, builds ``M`` bundled
    operators from ``c_ops_full``, and solves the master equation. The
    ensemble mean converges to the exact Lindblad result; the spread
    gives an honest error bar.

    ``H`` should already include any Lamb-shift contribution
    (``H_sys + H_LS``); the routine is otherwise solver-agnostic.

    Parameters
    ----------
    c_ops_full
        Full (unbundled) collapse-operator list -- typically the output
        of :func:`qutip_bundling.build_collapse_ops`.
    backend : {'qutip', 'native'}
        Master-equation propagator. ``'native'`` avoids building the
        Liouvillian and is required for large Hilbert spaces with many
        collapse operators.
    substeps
        For ``backend='native'``, number of RK4 substeps between each
        adjacent pair of times in ``tlist``. Increase for stiff systems.
    """
    e_ops = list(e_ops)
    if not e_ops:
        raise ValueError("mesolve_ensemble requires at least one e_op.")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    tlist = np.asarray(tlist)
    samples = np.empty((n_realizations, len(e_ops), tlist.size))
    for k in range(n_realizations):
        R = bundle(c_ops_full, M, distribution=distribution, rng=rng)
        samples[k] = _solve(H, rho0, tlist, R, e_ops, options, backend, substeps)

    mean = samples.mean(axis=0)
    if n_realizations > 1:
        std = samples.std(axis=0, ddof=1)
        sem = std / math.sqrt(n_realizations)
    else:
        std = np.zeros_like(mean)
        sem = np.zeros_like(mean)

    return BundledResult(
        times=tlist,
        expect=[mean[i] for i in range(len(e_ops))],
        sem=[sem[i] for i in range(len(e_ops))],
        std=[std[i] for i in range(len(e_ops))],
        samples=samples,
        M=M,
        n_realizations=n_realizations,
    )


def mesolve_jackknife(
    H,
    rho0,
    tlist,
    c_ops_full: Sequence[qutip.Qobj],
    M: int,
    e_ops: Sequence[qutip.Qobj],
    n_realizations: int = 16,
    distribution: str = "phase",
    rng: np.random.Generator | int | None = None,
    options=None,
    *,
    backend: str = "qutip",
    substeps: int = 1,
) -> BundledResult:
    """Bias-reduced bundled solve via the jackknife-2 estimator (eq. 16).

    Finite ``M`` introduces an O(1/M) bias because the dissipator noise
    enters the density matrix nonlinearly. For each realization this
    routine combines three solves built from the *same* pool of ``M``
    draws -- the full bundle and its two halves -- as

        O_jack2 = 2 * O_{1..M} - 0.5 * ( O_{1..M/2} + O_{M/2+1..M} )

    cancelling the leading bias. ``M`` must be even. The returned
    ``BundledResult`` holds the jackknife-corrected mean;
    ``extra["direct"]`` holds the uncorrected estimate for comparison.

    See :func:`mesolve_ensemble` for ``backend`` and ``substeps``.
    """
    e_ops = list(e_ops)
    if not e_ops:
        raise ValueError("mesolve_jackknife requires at least one e_op.")
    if M % 2 != 0:
        raise ValueError(f"jackknife requires even M, got {M}.")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    tlist = np.asarray(tlist)
    half = M // 2
    n_e = len(e_ops)
    direct = np.empty((n_realizations, n_e, tlist.size))
    jack = np.empty((n_realizations, n_e, tlist.size))

    for k in range(n_realizations):
        phases = random_phases(M, len(c_ops_full),
                                distribution=distribution, rng=rng)
        full = bundle_from_phases(c_ops_full, phases)
        first = bundle_from_phases(c_ops_full, phases[:half])
        second = bundle_from_phases(c_ops_full, phases[half:])

        o_full = _solve(H, rho0, tlist, full, e_ops, options, backend, substeps)
        o_first = _solve(H, rho0, tlist, first, e_ops, options, backend, substeps)
        o_second = _solve(H, rho0, tlist, second, e_ops, options, backend, substeps)

        direct[k] = o_full
        jack[k] = 2.0 * o_full - 0.5 * (o_first + o_second)

    mean = jack.mean(axis=0)
    direct_mean = direct.mean(axis=0)
    if n_realizations > 1:
        std = jack.std(axis=0, ddof=1)
        sem = std / math.sqrt(n_realizations)
    else:
        std = np.zeros_like(mean)
        sem = np.zeros_like(mean)

    return BundledResult(
        times=tlist,
        expect=[mean[i] for i in range(n_e)],
        sem=[sem[i] for i in range(n_e)],
        std=[std[i] for i in range(n_e)],
        samples=jack,                       # jackknife-corrected per-realization
        M=M,
        n_realizations=n_realizations,
        extra={
            "direct": [direct_mean[i] for i in range(n_e)],
            "direct_samples": direct,       # uncorrected per-realization
        },
    )
