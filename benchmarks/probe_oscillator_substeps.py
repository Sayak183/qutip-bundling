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
import argparse
import time
import numpy as np
import qutip

from common import build_davies_operators, build_oscillator_bath, TLIST
from qutip_bundling.native_solver import rk4_mesolve, SolverInstabilityError

NATIVE_REF_TOL = 1e-4          # same tolerance the benchmark certifies against
FOCK = {64: 32, 128: 64}      # dim -> n_fock (dim = 2 * n_fock)
SUBSTEP_LADDER = [8, 16, 32, 64, 128]


def certify(H, rho0, c_ops, substeps, tlist=TLIST):
    """Return (ref_time, self_check_dev, ok) or (time, inf, False) on divergence.

    ok means: the full-substep reference ran, AND halving the substeps changes
    the answer by <= NATIVE_REF_TOL (so the step size is fine, not marginal).
    """
    try:
        t0 = time.perf_counter()
        hi = rk4_mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[H], substeps=substeps)
        dt = time.perf_counter() - t0
    except SolverInstabilityError:
        return None, float("inf"), False
    try:
        lo = rk4_mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[H],
                         substeps=substeps // 2)
    except SolverInstabilityError:
        return dt, float("inf"), False
    dev = float(np.max(np.abs(np.real(hi.expect[0]) - np.real(lo.expect[0]))))
    return dt, dev, (np.isfinite(dev) and dev <= NATIVE_REF_TOL)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    ap.add_argument("--quick", type=float, nargs="?", const=0.5, default=None,
                    help="probe over the FIRST FIFTH of the time window "
                         "only. Fixed-step instability is a property of the "
                         "step size, not of how long you integrate, and it "
                         "shows up early (dim 64 blew up at t=1.03, dim 128 "
                         "at t=0.51) -- so this answers 'does it diverge?' "
                         "at ~1/5 the cost. A PASS here is necessary, not "
                         "sufficient: confirm over the full window before "
                         "trusting a reference.")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128],
                    help="which dimensions to probe")
    ap.add_argument("--substeps", type=int, nargs="+", default=SUBSTEP_LADDER,
                    help="ascending substep ladder to try")
    args = ap.parse_args()
    tlist = (TLIST[:max(2, int(len(TLIST) * args.quick))]
             if args.quick else TLIST)
    if args.quick:
        print(f"[quick {args.quick:g}] probing t = {tlist[0]:.3g} .. {tlist[-1]:.3g} "
              f"({len(tlist)} of {len(TLIST)} points)")
    print(f"{'dim':>5} {'N_L':>6} {'substeps':>9} {'ref_time':>10} "
          f"{'selfcheck_dev':>14} {'verdict':>9}")
    for dim in args.dims:
        n_fock = FOCK[dim]
        H, X, psi0 = build_oscillator_bath(n_fock)
        rho0 = qutip.ket2dm(psi0)
        c_ops = build_davies_operators(H, X)
        n_l = len(c_ops)
        first_ok = None
        for ss in args.substeps:
            dt, dev, ok = certify(H, rho0, c_ops, ss, tlist)
            tstr = "  (diverged)" if dt is None else f"{dt:9.1f}s"
            print(f"{dim:>5} {n_l:>6} {ss:>9} {tstr:>10} "
                  f"{dev:>14.2e} {'OK' if ok else 'FAIL':>9}")
            if ok:
                first_ok = ss
                break            # smallest working substep count found
        if first_ok is None:
            print(f"   -> dim {dim}: NOT certifiable up to {args.substeps[-1]} "
                  f"substeps (needs a stiff/implicit solver, not more RK4 steps)")
        else:
            factor = first_ok // 4
            print(f"   -> dim {dim}: first certifiable at {first_ok} substeps "
                  f"(~{factor}x the cost of the 4-substep points; must be a "
                  f"disclosed separate curve)")
        print()


if __name__ == "__main__":
    main()
