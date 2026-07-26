"""Jackknife rate strips (Result 6 validation), one per system.

For each system, collapses the per-dimension jackknife-rate convergence panels
(dim 16 / 32 / 64) into ONE shared-axis strip, so the "steepening resolves
where the sampling floor allows" story reads as a single small-to-large
comparison instead of loose PNGs.

Plot-only: reads the committed convergence_progress_*.json files that
benchmark_convergence.py already wrote -- no recompute. The per-panel fit uses
the SAME 2x-SEM / 3-point rule as benchmark_convergence.py, so the strips can
never disagree with the individual figures or the summary table.

Also prints the combined markdown table body BENCHMARKS.md embeds.

Run:  python plot_jackknife_rate_strip.py
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import add_settings_footer

# One strip per system, panels small-to-large. The unsuffixed progress file is
# dim 16 (original convergence run); _dim32/_dim64 are the larger sizes.
SYSTEMS = {
    "spin_chain": {
        "pretty": "Spin Chain",
        "panels": [
            ("convergence_progress_spin_chain.json",       "dim 16"),
            ("convergence_progress_spin_chain_dim32.json", "dim 32"),
            ("convergence_progress_spin_chain_dim64.json", "dim 64"),
        ],
    },
    "oscillator_bath": {
        "pretty": "Oscillator Bath",
        "panels": [
            ("convergence_progress_oscillator_bath.json",       "dim 16"),
            ("convergence_progress_oscillator_bath_dim32.json", "dim 32"),
            ("convergence_progress_oscillator_bath_dim64.json", "dim 64"),
        ],
    },
}
SEM_FACTOR = 2.0
MIN_POINTS = 3


def fit_stats(d):
    M    = np.asarray(d["M"], float)
    stat = np.asarray(d["stat"], float)
    bias = np.asarray(d["bias"], float)
    bjk  = np.asarray(d["bias_jk"], float)
    nr   = d["n_real"]
    sem  = stat / np.sqrt(nr)

    b_slope = np.polyfit(np.log(M), np.log(bias), 1)[0]
    above   = bjk > SEM_FACTOR * sem
    if above.sum() >= MIN_POINTS:
        jk_slope = np.polyfit(np.log(M[above]), np.log(bjk[above]), 1)[0]
        quotable = True
    else:
        jk_slope = float("nan")
        quotable = False

    red = bias / np.where(bjk > 0, bjk, np.nan)
    red_cleared = red[above]
    gain_lo = float(np.nanmin(red_cleared)) if above.any() else float("nan")
    gain_hi = float(np.nanmax(red_cleared)) if above.any() else float("nan")

    if quotable and jk_slope <= b_slope - 0.2:
        verdict = "rate steepens"
    elif above.sum() >= 1:
        verdict = "level only" if quotable else "marginal"
    else:
        verdict = "marginal"

    return dict(M=M, stat=stat, bias=bias, bjk=bjk, sem=sem, nr=nr,
                dim=d["dim"], n_l=d["n_l"], b_slope=b_slope, jk_slope=jk_slope,
                quotable=quotable, n_above=int(above.sum()),
                gain_lo=gain_lo, gain_hi=gain_hi, verdict=verdict)


def draw_panel(ax, f, label, show_ylabel):
    M, stat, bias, bjk, sem = f["M"], f["stat"], f["bias"], f["bjk"], f["sem"]
    ax.loglog(M, stat, "o-", color="tab:blue", lw=1.6, ms=5, label="spread")
    ax.loglog(M, sem, ":", color="tab:blue", alpha=0.5, label="SEM floor")
    ax.loglog(M, bias, "s-", color="tab:green", lw=1.6, ms=5,
              label=fr"bias, uncorr. ($M^{{{f['b_slope']:.2f}}}$)")
    jk_pos = bjk > 0
    jk_lab = (fr"bias, jackknife ($M^{{{f['jk_slope']:.2f}}}$)"
              if f["quotable"] else "bias, jackknife (upper bd.)")
    ax.loglog(M[jk_pos], bjk[jk_pos], "D-", color="tab:red", lw=1.6, ms=5,
              label=jk_lab)
    ax.loglog(M, bias[0]*(M/M[0])**-1.0, "--", color="tab:green", alpha=0.45)
    ax.loglog(M, bjk[0]*(M/M[0])**-2.0, "--", color="tab:red", alpha=0.45)

    ax.set_title(f"{label}  ($N_r$={f['nr']}, {f['n_above']} pts $>$2\u00b7SEM)",
                 fontsize=12)
    ax.set_xlabel("bundle size $M$", fontsize=12)
    if show_ylabel:
        ax.set_ylabel(r"max-over-time error in $\langle H\rangle$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(labelsize=10)


def build_strip(system_key, cfg):
    panels = cfg["panels"]
    fits = []
    for fname, label in panels:
        with open(fname) as fh:
            fits.append((fit_stats(json.load(fh)), label))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6*len(panels), 4.8),
                             sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for i, (ax, (f, label)) in enumerate(zip(axes, fits)):
        draw_panel(ax, f, label, show_ylabel=(i == 0))
    fig.suptitle(f"{cfg['pretty']}: jackknife bias correction vs $M$ "
                 f"(rate steepens where the sampling floor allows)",
                 fontsize=14, y=1.02)
    add_settings_footer(
        fig,
        "dashed guides: $M^{-1}$ (uncorrected) and $M^{-2}$ (jackknife target); "
        "solid-red slope fitted only above 2\u00b7SEM with \u22653 points, "
        "else reported as an upper bound",
        fontsize=11, wrap_chars=140,
    )
    out = f"benchmark_jackknife_rate_strip_{system_key}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return fits


def main():
    all_rows = []
    for system_key, cfg in SYSTEMS.items():
        fits = build_strip(system_key, cfg)
        for f, label in fits:
            all_rows.append((cfg["pretty"], label, f))
    print()

    print("| system | dim | uncorrected | jackknife | reduction | verdict |")
    print("|---|---|---|---|---|---|")
    for pretty, label, f in all_rows:
        jk = f"$M^{{{f['jk_slope']:.2f}}}$" if f["quotable"] else "\u2014"
        if np.isfinite(f["gain_lo"]):
            gain = (f"{f['gain_lo']:.1f}\u2013{f['gain_hi']:.1f}\u00d7"
                    if f["gain_hi"] > f["gain_lo"] else f"{f['gain_hi']:.1f}\u00d7")
        else:
            gain = "\u2014"
        print(f"| {pretty} | {label} | $M^{{{f['b_slope']:.2f}}}$ | {jk} | "
              f"{gain} | {f['verdict']} |")


if __name__ == "__main__":
    main()
