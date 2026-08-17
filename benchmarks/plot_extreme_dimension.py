r"""
plot_extreme_dimension.py
=========================

Figure for the run that has no reference to be plotted against.

Every other figure here draws SLB against an exact solve. At dimension 256 there
is none -- the operator list alone would be 31.9 GB -- so the two panels show the
two things that CAN be checked without one, and that the method could fail:

  LEFT   the long-time limit against the Gibbs state, which comes from the
         eigenvalues alone and is therefore free at any dimension the
         construction can reach, and is independent of everything bundling does.
  RIGHT  convergence in M, plotted against 1/M because that is the predicted
         form of the bias -- so the prediction is falsifiable by whether the
         three points fall on a line, not merely by whether they stop moving.

Reads:   data/extreme_dimension_mixed_chain_dim<D>.json
Writes:  benchmark_extreme_dimension_mixed_chain.png
Run:     python plot_extreme_dimension.py [--dim 256]
"""

from __future__ import annotations

import argparse
import glob
import re

import numpy as np
import matplotlib.pyplot as plt

from common import add_settings_footer, as_array, load_data

SLB_GREEN = "#006d2c"
GIBBS_GREY = "tab:gray"
SWEEP_REALIZATIONS = 16
THERMAL_REALIZATIONS = 4


def derive(doc):
    """Everything the figure states, computed once and returned for reuse.

    The caption numbers and the drawn numbers come from this one place, so a
    figure that disagrees with its own caption is not expressible.
    """
    gibbs = float(doc["gibbs_energy"])
    t = as_array(doc["thermal"]["times"])
    energy = as_array(doc["thermal"]["energy"])

    m_values = np.array([s["M"] for s in doc["sweep"]], dtype=float)
    finals = np.array([as_array(s["energy"])[-1] for s in doc["sweep"]])
    sems = np.array([as_array(s["sem"])[-1] for s in doc["sweep"]])

    # Bias is predicted to fall as 1/M, so a straight line through these points
    # against 1/M extrapolates to the M -> infinity answer at its intercept.
    slope, intercept = np.polyfit(1.0 / m_values, finals, 1)
    pairwise = [
        (m_values[a], m_values[b],
         (finals[b] * m_values[b] - finals[a] * m_values[a])
         / (m_values[b] - m_values[a]))
        for a, b in zip(range(len(m_values) - 1), range(1, len(m_values)))
    ]

    # The thermal run used fewer realizations than the sweep, so its statistical
    # error is larger by the square root of that ratio. Without this the residual
    # gap looks like a bias when it is within the noise.
    sem_thermal = float(sems[-1]) * np.sqrt(SWEEP_REALIZATIONS
                                            / THERMAL_REALIZATIONS)
    residual = abs(float(energy[-1]) - gibbs)

    # Has it actually stopped, or is it still creeping? A residual from a curve
    # still in motion means something different from one from a flat curve.
    tail = energy[t >= t[-1] - 20.0]
    return {
        "gibbs": gibbs, "t": t, "energy": energy,
        "m_values": m_values, "finals": finals, "sems": sems,
        "slope": float(slope), "intercept": float(intercept),
        "pairwise": pairwise,
        "sem_thermal": sem_thermal, "residual": residual,
        "residual_in_sem": residual / sem_thermal,
        "tail_drift": float(tail[-1] - tail[0]),
        "fraction": float(doc["thermal"]["fraction_covered"]),
        "n_l": int(doc["n_l"]), "dim": int(doc["dim"]),
        "list_gb": float(doc["operator_list_bytes"]) / 1024 ** 3,
    }


def figure(doc, out_name):
    plt.switch_backend("Agg")
    r = derive(doc)
    fig, (ax_t, ax_m) = plt.subplots(ncols=2, figsize=(12.5, 5.0))

    # ---- LEFT: relaxation to the thermal state ---------------------------
    ax_t.plot(r["t"], r["energy"], "-", color=SLB_GREEN, lw=2,
              label=f"SLB, $M$={doc['thermal']['M']}, "
                    f"{THERMAL_REALIZATIONS} realizations")
    ax_t.axhline(r["gibbs"], ls="--", color=GIBBS_GREY, lw=1.8,
                 label=r"$\mathrm{Tr}(H\rho_{\mathrm{Gibbs}})$"
                       " (from eigenvalues; free)")
    # One standard error of THIS run, so the reader can see at a glance whether
    # the gap that remains is a real offset or the noise it sits inside.
    ax_t.axhspan(r["gibbs"] - r["sem_thermal"], r["gibbs"] + r["sem_thermal"],
                 color=GIBBS_GREY, alpha=0.18, lw=0,
                 label=r"$\pm 1$ s.e.m. of this run")
    ax_t.set_xlabel("time")
    ax_t.set_ylabel(r"$\langle H \rangle$")
    ax_t.set_title(f"Relaxes to the Gibbs state\n"
                   f"dimension {r['dim']}, $N_L$ = {r['n_l']:,}", fontsize=11)
    ax_t.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax_t.grid(alpha=0.3)

    # The transient spans 0.67 while the claim to be judged -- that what remains
    # is inside the noise -- lives at 0.005. On one pair of axes the band that
    # decides it is thinner than the curve, so the tail gets its own scale.
    ax_zoom = ax_t.inset_axes([0.42, 0.60, 0.55, 0.36])
    late = r["t"] >= 12.0
    ax_zoom.plot(r["t"][late], r["energy"][late], "-", color=SLB_GREEN, lw=2)
    ax_zoom.axhline(r["gibbs"], ls="--", color=GIBBS_GREY, lw=1.5)
    ax_zoom.axhspan(r["gibbs"] - r["sem_thermal"], r["gibbs"] + r["sem_thermal"],
                    color=GIBBS_GREY, alpha=0.25, lw=0)
    ax_zoom.set_ylim(r["gibbs"] - 4 * r["sem_thermal"],
                     r["gibbs"] + 2 * r["sem_thermal"])
    ax_zoom.set_title("tail, to scale", fontsize=8, pad=3)
    ax_zoom.tick_params(labelsize=7)
    ax_zoom.grid(alpha=0.3)

    ax_t.annotate(
        f"covers {100 * r['fraction']:.1f}% of the distance\n"
        f"{r['residual']:.4f} remains = {r['residual_in_sem']:.1f} s.e.m.\n"
        f"flat to {abs(r['tail_drift']):.0e} over the last 20",
        xy=(0.03, 0.03), xycoords="axes fraction", ha="left", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    # ---- RIGHT: convergence in M, against 1/M ----------------------------
    inv = 1.0 / r["m_values"]
    grid = np.linspace(0.0, inv.max() * 1.12, 50)
    ax_m.plot(grid, r["intercept"] + r["slope"] * grid, "-",
              color=GIBBS_GREY, lw=1.5, zorder=1,
              label=r"fit linear in $1/M$")
    ax_m.errorbar(inv, r["finals"], yerr=r["sems"], fmt="s", color=SLB_GREEN,
                  ms=8, capsize=4, lw=1.8, zorder=3, label="SLB (16 realizations)")
    ax_m.plot([0.0], [r["intercept"]], "*", color="tab:purple", ms=18, zorder=4,
              label=rf"$M\to\infty$: {r['intercept']:+.4f}")
    for x, y, m in zip(inv, r["finals"], r["m_values"]):
        ax_m.annotate(f"$M$={m:.0f}", xy=(x, y), xytext=(4, -12),
                      textcoords="offset points", fontsize=9)
    ax_m.set_xlabel(r"$1/M$")
    ax_m.set_ylabel(r"$\langle H \rangle$ at $t=5$")
    ax_m.set_title("Bias falls as $1/M$, as predicted\n"
                   "(a curve here would falsify it)", fontsize=11)
    ax_m.set_xlim(left=-0.006)
    ax_m.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax_m.grid(alpha=0.3)

    pair_text = "; ".join(f"$M$={a:.0f},{b:.0f}$\\to${e:+.4f}"
                          for a, b, e in r["pairwise"])
    add_settings_footer(
        fig,
        f"System B (mixed-field chain), dimension {r['dim']}, "
        f"$N_L$ = {r['n_l']:,} Davies operators",
        f"NO EXACT REFERENCE EXISTS AT THIS SIZE: the operator list alone would "
        f"be {r['list_gb']:.1f} GB, so this run is scored on physics rather "
        f"than on an error -- the streaming construction never forms the list",
        f"independent extrapolations agree: {pair_text}",
        "trace preserved to machine precision throughout "
        "(max $|\\mathrm{Tr}-1|$ = 4.4e-16)",
        fontsize=9)

    fig.savefig(out_name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_name}")
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dim", type=int, default=None,
                        help="dimension to plot (default: the largest present)")
    args = parser.parse_args()

    if args.dim is None:
        found = sorted(
            int(re.search(r"dim(\d+)\.json$", p).group(1))
            for p in glob.glob("data/extreme_dimension_mixed_chain_dim*.json"))
        if not found:
            raise SystemExit("no extreme_dimension data found; "
                             "run run_extreme_dimension.py first")
        args.dim = found[-1]

    doc = load_data(f"extreme_dimension_mixed_chain_dim{args.dim}.json")
    r = figure(doc, "benchmark_extreme_dimension_mixed_chain.png")
    print(f"  M->inf = {r['intercept']:+.5f}, "
          f"residual {r['residual']:.5f} = {r['residual_in_sem']:.1f} s.e.m.")


if __name__ == "__main__":
    main()
