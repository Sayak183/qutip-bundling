"""
probe_oscillator_substeps.py
============================
One-shot diagnostic: at which substep count does the native full-dissipator
reference for the OSCILLATOR become CERTIFIABLE at dim 64 and dim 128?

The oscillator's anharmonic ladder is stiff (frequencies ~ n^2), so fixed-step
RK4 diverges at the default 4 substeps beyond dim 32. Raising substeps shrinks
the step and restores stability. This probe finds the smallest substep count
whose reference passes the same self-check the benchmark uses: recompute at half
the substeps and require the two to agree to NATIVE_REF_TOL.

It touches NOTHING in the repo -- it only prints a table you can read off:

    dim   substeps   ref_time   self-check dev    verdict
     64          8      ...s          1.9e+92      FAIL
     64         16      ...s          3e-09        OK  <- smallest that works
     ...

Run:  python probe_oscillator_substeps.py
"""
from __future__ import annotations
import time
import numpy as np
import qutip

from common import gamma, build_oscillator_bath, TLIST
from qutip_bundling import davies_operators
from qutip_bundling.native_solver import rk4_mesolve, SolverInstabilityError

NATIVE_REF_TOL = 1e-4          # same tolerance the benchmark certifies against
FOCK = {64: 32, 128: 64}      # dim -> n_fock (dim = 2 * n_fock)
SUBSTEP_LADDER = [8, 16, 32, 64]


def certify(H, rho0, c_ops, substeps):
    """Return (ref_time, self_check_dev, ok) or (time, inf, False) on divergence.

    ok means: the full-substep reference ran, AND halving the substeps changes
    the answer by <= NATIVE_REF_TOL (so the step size is fine, not marginal).
    """
    try:
        t0 = time.perf_counter()
        hi = rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H], substeps=substeps)
        dt = time.perf_counter() - t0
    except SolverInstabilityError:
        return None, float("inf"), False
    try:
        lo = rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H],
                         substeps=substeps // 2)
    except SolverInstabilityError:
        return dt, float("inf"), False
    dev = float(np.max(np.abs(np.real(hi.expect[0]) - np.real(lo.expect[0]))))
    return dt, dev, (np.isfinite(dev) and dev <= NATIVE_REF_TOL)


def main():
    print(f"{'dim':>5} {'N_L':>6} {'substeps':>9} {'ref_time':>10} "
          f"{'selfcheck_dev':>14} {'verdict':>9}")
    for dim in (64, 128):
        n_fock = FOCK[dim]
        H, X, psi0 = build_oscillator_bath(n_fock)
        rho0 = qutip.ket2dm(psi0)
        c_ops = davies_operators(H, X, gamma)
        n_l = len(c_ops)
        first_ok = None
        for ss in SUBSTEP_LADDER:
            dt, dev, ok = certify(H, rho0, c_ops, ss)
            tstr = "  (diverged)" if dt is None else f"{dt:9.1f}s"
            print(f"{dim:>5} {n_l:>6} {ss:>9} {tstr:>10} "
                  f"{dev:>14.2e} {'OK' if ok else 'FAIL':>9}")
            if ok:
                first_ok = ss
                break            # smallest working substep count found
        if first_ok is None:
            print(f"   -> dim {dim}: NOT certifiable up to {SUBSTEP_LADDER[-1]} "
                  f"substeps (needs a stiff/implicit solver, not more RK4 steps)")
        else:
            factor = first_ok // 4
            print(f"   -> dim {dim}: first certifiable at {first_ok} substeps "
                  f"(~{factor}x the cost of the 4-substep points; must be a "
                  f"disclosed separate curve)")
        print()


if __name__ == "__main__":
    main()
