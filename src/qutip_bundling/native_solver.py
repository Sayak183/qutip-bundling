"""Native RK4 master equation propagator.

QuTiP's ``mesolve`` builds the full Liouvillian superoperator -- an
N^2 x N^2 matrix for each collapse operator -- which is wasteful when
the number of collapse operators is large, and outright impossible
once N gets into the dozens.

This module steps

    drho/dt = -i [H, rho] + sum_alpha ( L_alpha rho L_alpha^dag
                                         - 0.5 { L_alpha^dag L_alpha, rho } )

directly with classical RK4 in operator space. Cost per step is

    O( N_c * N^3 + N^3 )

with O(N^2) extra storage. No N^4 anywhere.

The propagator is solver-agnostic about its inputs; the bundling code
just hands it a list of (possibly bundled) collapse operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import qutip


__all__ = ["rk4_mesolve", "NativeResult", "SolverInstabilityError"]


# Guard thresholds for the per-point divergence check in ``rk4_mesolve``.
# A valid density matrix has ``|rho_ij| <= Tr(rho)``; entries three orders of
# magnitude beyond that are unambiguous divergence long before float overflow
# ("soft divergence" -- e.g. entries of ~1e97 are finite but garbage, and once
# slipped past a finiteness-only check). Trace drift is a second, independent
# symptom: the generator is trace-free, so RK4 conserves the trace to rounding
# and any real drift means numerical breakdown.
_SOFT_DIVERGENCE_FACTOR = 1e3
_TRACE_DRIFT_TOL = 1e-6


class SolverInstabilityError(RuntimeError):
    """Raised when the RK4 integration diverges to a non-finite state.

    The fixed-step RK4 propagator is only conditionally stable: if the Lindblad
    generator is too stiff for the chosen step size, the density matrix grows
    without bound and overflows to inf/NaN. Rather than return a silently
    corrupted result, ``rk4_mesolve`` raises this so the caller can increase
    ``substeps`` (or refine ``tlist``).
    """


@dataclass
class NativeResult:
    """Mimics the relevant attributes of ``qutip.solver.Result``."""
    times: np.ndarray
    expect: list[np.ndarray]
    states: list | None = None     # full Qobj density matrices if requested


def _to_array(op: qutip.Qobj) -> np.ndarray:
    return np.asarray(op.full(), dtype=np.complex128)


def _dissipator_rhs(rho: np.ndarray,
                    Heff: np.ndarray,
                    Heff_dag: np.ndarray,
                    C_list: list[np.ndarray],
                    Cd_list: list[np.ndarray]) -> np.ndarray:
    """Compute drho/dt for a Lindblad master equation.

    Uses the precomputed non-Hermitian effective Hamiltonian
    ``Heff = -i H - 0.5 sum_k C_k^dag C_k`` to combine the coherent
    commutator and anti-commutator into a single pair of matrix multiplications:
        Heff @ rho + rho @ Heff_dag
    followed by the sandwich terms sum_k C_k @ rho @ C_k^dag.

    Parameters
    ----------
    rho      : (N, N) complex array, density matrix at this instant.
    Heff     : (N, N) complex array, -i H - 0.5 CdC_sum.
    Heff_dag : (N, N) complex array,  i H - 0.5 CdC_sum.
    C_list   : list of K dense (N, N) collapse-operator arrays.
    Cd_list  : list of their adjoints.
    """
    out = Heff @ rho + rho @ Heff_dag
    for C_k, Cd_k in zip(C_list, Cd_list):
        out += C_k @ rho @ Cd_k
    return out


def rk4_mesolve(
    H: qutip.Qobj,
    rho0: qutip.Qobj,
    tlist: Sequence[float] | np.ndarray,
    c_ops: Sequence[qutip.Qobj],
    e_ops: Sequence[qutip.Qobj] | None = None,
    *,
    substeps: int = 1,
    store_states: bool = False,
) -> NativeResult:
    """Classical RK4 stepper for the Lindblad master equation.

    Memory: O((K + a few) * N^2) where K = len(c_ops). No N^4 superoperator
    is ever built, so this reaches Hilbert dim ~100+ on a 4 GB machine
    where ``qutip.mesolve`` cannot run.

    Cost per step: O(K * N^3) for the dissipator plus O(N^3) for the
    coherent part and the precomputed anti-commutator.

    The Lindblad generator is trace-free, so RK4 conserves ``Tr(rho)`` to
    machine precision, and each substep re-Hermitizes ``rho``. Positivity is
    not explicitly projected -- it holds only to the integrator's order, which
    is why stiff systems need enough ``substeps`` (a single substep can
    diverge). Divergence is checked at every ``tlist`` point -- not only when
    the state overflows to non-finite values, but as soon as matrix entries
    grow far beyond the physical bound ``|rho_ij| <= Tr(rho)`` or the
    (exactly conserved) trace drifts, so a large-but-finite runaway state is
    caught early rather than slipping through a finiteness-only check. A
    ``SolverInstabilityError`` is raised instead of returning a silently
    corrupted result. Validate against ``qutip.mesolve`` on a small system.

    Expectation values are recorded as real (observables assumed Hermitian).

    Parameters
    ----------
    H, rho0, tlist, c_ops, e_ops
        Same meaning as in ``qutip.mesolve``.
    substeps : int, default 1
        Number of RK4 substeps between adjacent ``tlist`` points.
        Stiffer problems (more or larger collapse operators) need more
        substeps. A reasonable starting heuristic is
        ``substeps ~= max(1, int(K * dt * gamma_max))``.
    store_states : bool, default False
        If True, save the full density matrix at every time point.
    """
    tlist = np.asarray(tlist, dtype=float)
    if tlist.size < 2:
        raise ValueError("tlist must have at least 2 entries.")
    if rho0.isket:
        rho0 = qutip.ket2dm(rho0)
    rho = _to_array(rho0)
    N = rho.shape[0]
    H_arr = _to_array(H)

    C_list = [_to_array(c) for c in c_ops]
    Cd_list = [np.conj(C).T for C in C_list]
    if C_list:
        CdC_sum = np.zeros((N, N), dtype=np.complex128)
        for C_k, Cd_k in zip(C_list, Cd_list):
            CdC_sum += Cd_k @ C_k
    else:
        CdC_sum = np.zeros((N, N), dtype=np.complex128)

    Heff = -1j * H_arr - 0.5 * CdC_sum
    Heff_dag = 1j * H_arr - 0.5 * CdC_sum

    e_ops = list(e_ops) if e_ops else []
    e_arrs = [_to_array(op) for op in e_ops]
    expect = [np.zeros(tlist.size, dtype=float) for _ in e_arrs]
    states_out: list = [] if store_states else None

    def record(i, rho_arr):
        for j, eop in enumerate(e_arrs):
            expect[j][i] = float(np.real(np.trace(eop @ rho_arr)))
        if store_states:
            states_out.append(qutip.Qobj(rho_arr.copy(), dims=rho0.dims))

    tr0 = float(np.trace(rho).real)
    _scale = max(1.0, abs(tr0))
    entry_bound = _SOFT_DIVERGENCE_FACTOR * _scale
    trace_tol = _TRACE_DRIFT_TOL * _scale

    record(0, rho)
    for i in range(tlist.size - 1):
        dt = (tlist[i + 1] - tlist[i]) / substeps
        for _ in range(substeps):
            k1 = _dissipator_rhs(rho,                 Heff, Heff_dag, C_list, Cd_list)
            k2 = _dissipator_rhs(rho + 0.5 * dt * k1, Heff, Heff_dag, C_list, Cd_list)
            k3 = _dissipator_rhs(rho + 0.5 * dt * k2, Heff, Heff_dag, C_list, Cd_list)
            k4 = _dissipator_rhs(rho + dt * k3,       Heff, Heff_dag, C_list, Cd_list)
            rho = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            rho = 0.5 * (rho + rho.conj().T)
        if not np.isfinite(rho).all():
            symptom = "the density matrix became non-finite"
        else:
            max_abs = float(np.abs(rho).max())
            tr_drift = abs(float(np.trace(rho).real) - tr0)
            if max_abs > entry_bound:
                symptom = (f"matrix entries grew to {max_abs:.3g} while still "
                           f"finite (soft divergence; a physical state has "
                           f"entries <= Tr(rho) = {tr0:.3g})")
            elif tr_drift > trace_tol:
                symptom = (f"Tr(rho) drifted by {tr_drift:.3g} (the generator "
                           f"is trace-free, so any drift is numerical "
                           f"breakdown)")
            else:
                symptom = None
        if symptom is not None:
            raise SolverInstabilityError(
                f"rk4_mesolve diverged near t={tlist[i + 1]:.4g} with "
                f"substeps={substeps}: {symptom}. "
                f"The Lindblad generator is too stiff for this step size -- "
                f"increase 'substeps' (e.g. try {max(substeps, 1) * 2}) or use a "
                f"finer tlist spacing."
            )
        record(i + 1, rho)

    return NativeResult(times=tlist, expect=expect, states=states_out)
