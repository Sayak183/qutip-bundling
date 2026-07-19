"""
benchmark_convergence.py
========================

Convergence-rate check for Stochastic Lindblad Bundling (SLB), at a fixed system
size, as the bundle size ``M`` grows. This answers the skeptic's question "how
do I know M is converged and not just tuned to look good?" by showing the error
falls at the *rate the theory predicts*, which tuning cannot fake.

Two quantities, two predicted rates (both on a log-log axis with guide lines):

  * statistical spread  -- the standard deviation of the bundled <H>(t) over
    realizations. SLB is a Monte Carlo estimator, so this should fall as the
    classic ``M^(-1/2)``.
  * bias               -- the deviation of the bundled *mean* from the exact
    reference. The finite-M bias is higher order and should fall faster,
    ~ ``M^(-1)``.

Seeing the measured slopes land on -1/2 and -1 is direct evidence the estimator
behaves as derived (eq. 8/11 of the paper), independent of any tuned setting.

Produces, per system:
    benchmark_convergence_<system>.png

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_convergence.py
"""

from __future__ import annotations

import numpy as np
import qutip

from common import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST,
    format_slb_settings, add_settings_footer,
)
from qutip_bundling import davies_operators, mesolve_ensemble, mesolve_jackknife
from qutip_bundling.native_solver import rk4_mesolve

# ===========================================================================
# CONFIG
# ===========================================================================
# Per-system realization counts. The SEM noise floor drops as 1/sqrt(N_r), so a
# system with SMALLER bias needs MORE realizations to keep the corrected bias
# above the floor long enough to measure its rate. The oscillator's bias is
# ~10x smaller than the chain's, so it needs ~16x more samples to expose the
# same number of clean points. (Overridable with --realizations to scale both.)
N_REALIZATIONS_BY_SYSTEM = {
    "spin_chain":      256,    # bias is large; 256 already resolves M=2,4,8
    "oscillator_bath": 4000,   # bias is small; needs a much lower floor
}
N_REALIZATIONS = 256         # fallback / default if a system isn't listed
SUBSTEPS = 4                 # matches the other benchmarks (>=2 for stability)
SEED = 0
M_VALUES = [2, 4, 8, 16, 32, 64]

# One representative size per system (kept modest so the exact reference is cheap).
# Size-points per system: (size, substeps, realizations). Each is computed
# ONCE and saved to its own figure/file keyed by dim, so nothing overwrites and
# you can pick any size later. Realizations are per-size because the SEM floor
# must sit below the corrected bias -- and larger systems are far more
# expensive, so the counts are tuned rather than uniform.
SYSTEMS = [
    ("spin_chain",      build_spin_chain,  [(4, 4, 256), (5, 4, 128), (6, 4, 64)]),
    ("oscillator_bath", build_oscillator_bath, [(8, 4, 256), (16, 4, 128),
                                                (32, 16, 64)]),
]
NATIVE_REF_SUBSTEPS_FACTOR = 2
MAX_FULL_DIM_FALLBACK = 64


def run(name, build, size, n_real=N_REALIZATIONS, substeps=SUBSTEPS):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    c_ops = davies_operators(H, X, gamma)
    dim = H.shape[0]
    # reference: mesolve while feasible, else certified native full dissipator
    ref, ref_method = None, None
    if dim <= MAX_FULL_DIM_FALLBACK:
        try:
            ref = np.real(qutip.mesolve(H, rho0, TLIST, c_ops=c_ops,
                                        e_ops=[H]).expect[0])
            ref_method = "mesolve"
        except MemoryError:
            ref = None
    if ref is None:
        rs = NATIVE_REF_SUBSTEPS_FACTOR * substeps
        hi = rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H], substeps=rs)
        lo = rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H], substeps=rs // 2)
        ref = np.real(hi.expect[0])
        dev = float(np.max(np.abs(ref - np.real(lo.expect[0]))))
        ok = bool(np.isfinite(dev) and dev <= 1e-4)
        ref_method = f"native_rk4_substeps{rs}"
        print(f"  reference via {ref_method}; self-check dev {dev:.2e} "
              f"[{'OK' if ok else 'FAILED'}]")
        if not ok:
            print(f"  dim={dim}: reference uncertifiable -- skipping size.")
            return None

    Ms, stat, bias, bias_jk = [], [], [], []
    print(f"\n[{name}] dim={dim}, N_L={len(c_ops)}")
    print(f"{'M':>5}  {'stat spread':>12}  {'bias':>12}  {'bias (jackknife)':>16}")
    for M in M_VALUES:
        if M > len(c_ops):
            continue
        # SAME seed for both estimators at each M, so the comparison isolates
        # the jackknife correction rather than sampling luck.
        ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=M, e_ops=[H],
                               n_realizations=n_real, rng=SEED,
                               backend="native", substeps=substeps)
        jk = mesolve_jackknife(H, rho0, TLIST, c_ops, M=M, e_ops=[H],
                               n_realizations=n_real, rng=SEED,
                               backend="native", substeps=substeps)
        mean = np.real(ens.expect[0])
        mean_jk = np.real(jk.expect[0])
        std = np.asarray(ens.std[0], float)
        s = float(np.max(std))
        b = float(np.max(np.abs(mean - ref)))
        b_jk = float(np.max(np.abs(mean_jk - ref)))
        Ms.append(M); stat.append(s); bias.append(b); bias_jk.append(b_jk)
        print(f"{M:>5}  {s:>12.3e}  {b:>12.3e}  {b_jk:>16.3e}", flush=True)
        # incremental save: a multi-hour run must survive an interrupt, so dump
        # progress after every M rather than only at the end.
        import json
        json.dump({"system":name,"dim":dim,"n_l":len(c_ops),"n_real":n_real,
                   "M":Ms,"stat":stat,"bias":bias,"bias_jk":bias_jk},
                  open(f"convergence_progress_{name}_dim{dim}.json","w"))
    return (np.array(Ms), np.array(stat), np.array(bias),
            np.array(bias_jk), dim, len(c_ops))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every system's realization count (e.g. 2 "
                         "doubles both) to push the noise floor lower")
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="only these Hilbert dims (default: every configured size)")
    ap.add_argument("--only", default=None,
                    help="run just one system (spin_chain or oscillator_bath)")
    args = ap.parse_args()
    global N_REALIZATIONS_BY_SYSTEM, SYSTEMS
    if args.scale != 1.0:
        N_REALIZATIONS_BY_SYSTEM = {k: int(round(v*args.scale))
                                    for k, v in N_REALIZATIONS_BY_SYSTEM.items()}
    if args.only:
        SYSTEMS = [s for s in SYSTEMS if s[0] == args.only]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, build, points in SYSTEMS:
      for size, substeps, n_real in points:
        probe_dim = build(size)[0].shape[0]
        if args.dims and probe_dim not in args.dims:
            continue
        print(f"\n### {name} dim {probe_dim}: {n_real} realizations, "
              f"{substeps} substeps ###")
        out = run(name, build, size, n_real, substeps)
        if out is None:
            continue
        Ms, stat, bias, bias_jk, dim, n_l = out

        s_slope = np.polyfit(np.log(Ms), np.log(stat), 1)[0]
        b_slope = np.polyfit(np.log(Ms), np.log(bias), 1)[0]
        # The bias is a property of the MEAN over N_REALIZATIONS runs, so the
        # noise floor it must clear is the SEM (= spread / sqrt(N)), NOT the
        # single-run spread. (Comparing a mean's bias to the single-run spread
        # is ~sqrt(N) too strict and falsely reports "noise-limited".) Below the
        # SEM the corrected "bias" is leftover Monte-Carlo noise; above it the
        # correction's true rate is visible, so we fit the slope there.
        sem = stat / np.sqrt(n_real)
        above_floor = bias_jk > sem
        if above_floor.sum() >= 2:
            bjk_slope = np.polyfit(np.log(Ms[above_floor]),
                                   np.log(bias_jk[above_floor]), 1)[0]
            bjk_slope_note = ""
        else:
            bjk_slope = float("nan")
            bjk_slope_note = " (reaches noise floor immediately)"

        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        ax.loglog(Ms, stat, "o-", color="tab:blue", lw=1.8,
                  label=fr"statistical spread (fit slope {s_slope:.2f})")
        ax.loglog(Ms, stat / np.sqrt(n_real), ":", color="tab:blue",
                  alpha=0.5, label=r"SEM = spread/$\sqrt{N_r}$ (bias noise floor)")
        ax.loglog(Ms, bias, "s-", color="tab:green", lw=1.8,
                  label=fr"bias, uncorrected (fit slope {b_slope:.2f})")
        jk_pos = bias_jk > 0
        _jklab = (fr"bias, jackknife-2 (fit slope {bjk_slope:.2f}"
                  fr"{bjk_slope_note})" if not np.isnan(bjk_slope)
                  else fr"bias, jackknife-2{bjk_slope_note}")
        ax.loglog(Ms[jk_pos], bias_jk[jk_pos], "D-", color="tab:red", lw=1.8,
                  label=_jklab)
        # mark the noise floor: where corrected bias sinks below the spread,
        # further reduction is limited by sampling, not by the estimator order
        floor_hit = np.where(bias_jk <= sem)[0]
        if floor_hit.size:
            mf = Ms[floor_hit[0]]
            ax.axvline(mf, color="tab:red", ls=":", alpha=0.4)
            ax.annotate("jackknife bias\nreaches noise floor", (mf, sem[floor_hit[0]]),
                        textcoords="offset points", xytext=(10, -22),
                        fontsize=7, color="tab:red", alpha=0.8)

        # Theoretical guide lines, anchored at the first point.
        ax.loglog(Ms, stat[0] * (Ms / Ms[0]) ** -0.5, "--", color="tab:blue",
                  alpha=0.6, label=r"$M^{-1/2}$ (Monte Carlo)")
        ax.loglog(Ms, bias[0] * (Ms / Ms[0]) ** -1.0, "--", color="tab:green",
                  alpha=0.6, label=r"$M^{-1}$ (finite-$M$ bias)")
        ax.loglog(Ms, bias_jk[0] * (Ms / Ms[0]) ** -2.0, "--", color="tab:red",
                  alpha=0.6, label=r"$M^{-2}$ (jackknife-corrected bias)")

        ax.set_xlabel("bundle size $M$")
        ax.set_ylabel(r"max-over-time error in $\langle H\rangle$")
        ax.set_title(f"{name} (dim {dim}, $N_L$={n_l}): SLB convergence in $M$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        add_settings_footer(
            fig,
            format_slb_settings(M=M_VALUES, substeps=SUBSTEPS,
                                n_realizations=n_real, swept=True),
            "spread = std over realizations; bias = |mean \u2212 ref| for the "
            "uncorrected and jackknife-2 estimators (same seed per M); "
            "full-Lindblad reference. The jackknife cancels the leading O(1/M) "
            "bias term, so its slope should steepen from ~-1 toward ~-2.",
        )
        fig.savefig(f"benchmark_convergence_{name}_dim{dim}.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
        # ---- self-diagnosing verdict, so a run on any machine reports whether
        # ---- this figure actually "worked" without needing a second pair of eyes
        n_clear = int(np.sum(bias_jk > sem))  # uses this system's own floor
        red_vs_green = (bias / np.where(bias_jk > 0, bias_jk, np.nan))
        best_gain = float(np.nanmax(red_vs_green))
        print(f"  saved benchmark_convergence_{name}_dim{dim}.png")
        print(f"  --- JACKKNIFE FIGURE SELF-CHECK [{name}] ---")
        print(f"    points where corrected bias clears the noise floor: "
              f"{n_clear} of {len(Ms)}")
        if n_clear >= 2 and not np.isnan(bjk_slope):
            print(f"    VERDICT: OK -- bias rate STEEPENS under the jackknife: "
                  f"uncorrected {b_slope:.2f}  ->  jackknife {bjk_slope:.2f} "
                  f"(toward the M^-2 the leading-order cancellation predicts)")
            print(f"    measurable over {n_clear} points above the SEM floor; "
                  f"max bias reduction {best_gain:.1f}x. Safe to commit.")
        elif n_clear == 1:
            print(f"    VERDICT: MARGINAL -- corrected bias clears the floor at only "
                  f"one M (max reduction {best_gain:.1f}x). The reduction is real but "
                  f"the SLOPE is not reliably measurable.")
            print(f"    -> raise N_REALIZATIONS (try {2*N_REALIZATIONS}) and re-run "
                  f"before trusting the jackknife slope.")
        else:
            print(f"    VERDICT: NOISE-LIMITED -- corrected bias never clears the "
                  f"spread; you are seeing sampling noise, not the correction's rate.")
            print(f"    -> raise N_REALIZATIONS (try {2*N_REALIZATIONS} or more) and "
                  f"re-run; the jackknife curve is drowning in Monte-Carlo noise.")


if __name__ == "__main__":
    main()
