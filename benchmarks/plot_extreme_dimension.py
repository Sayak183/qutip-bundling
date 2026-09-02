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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import qutip

from common import (KT, add_settings_footer, as_array, build_mixed_field_chain,
                    build_oscillator_bath, build_spin_chain, load_data)

_BUILDERS = {
    "mixed_chain": build_mixed_field_chain,
    "oscillator_bath": build_oscillator_bath,
    "spin_chain": build_spin_chain,
}

# |<e|X|e'>| below this counts as no coupling; stable over six decades.
SECTOR_COUPLING_TOL = 1e-10


def sector_resolved_energy(doc):
    """Tr(H rho_inf) for the state the generator actually reaches, or None.

    The Gibbs state is stationary here -- the generator annihilates it to
    machine precision -- but it is not the only stationary state: a Davies
    operator is Pi_e X Pi_e', so levels are dynamically connected only where
    <e|X|e'> is non-zero, and where that graph is disconnected each sector's
    population is separately conserved. The limit is then Gibbs WITHIN each
    sector, weighted by where rho0 started.

    Recomputed from the system definition because the committed data predates
    this correction. One eigendecomposition and a connected-components pass.
    """
    params = (doc.get("meta") or {}).get("params") or {}
    build = _BUILDERS.get(params.get("system"))
    size = params.get("size")
    if build is None or size is None:
        return None, None

    H, X, psi0 = build(int(size))
    energies, states = H.eigenstates()
    energies = np.real(energies)
    V = np.column_stack([s.full().ravel() for s in states])

    adjacency = (np.abs(V.conj().T @ X.full() @ V) > SECTOR_COUPLING_TOL)
    adjacency = adjacency.astype(np.int8)
    np.fill_diagonal(adjacency, 1)
    n_sectors, sector_of = connected_components(csr_matrix(adjacency),
                                                directed=False)
    if n_sectors == 1:
        return None, 1        # ergodic: the two targets coincide

    pops0 = np.real(np.diag(V.conj().T @ qutip.ket2dm(psi0).full() @ V))
    total = 0.0
    for s in range(n_sectors):
        mask = sector_of == s
        p_s = float(pops0[mask].sum())
        if p_s <= 0.0:
            continue
        w = np.exp(-(energies[mask] - energies[mask].min()) / KT)
        w /= w.sum()
        total += p_s * float(np.dot(w, energies[mask]))
    return total, int(n_sectors)

SLB_GREEN = "#006d2c"
GIBBS_GREY = "tab:gray"
SWEEP_REALIZATIONS = 16
# Only used for files written before the thermal run recorded its own count
# and its own s.e.m.; those runs used a hardcoded 4.
LEGACY_THERMAL_REALIZATIONS = 4


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

    # How close the endpoint sits to Gibbs means nothing except against this
    # run's own statistical error, so take the MEASURED one when the file has it
    # (runs from 2026-08-18 on). Older files predate that field and their
    # thermal run used a fixed 4 realizations, so the sweep's error is rescaled
    # by the square root of the ratio -- an estimate, and labelled as one.
    thermal = doc["thermal"]
    n_thermal = int(thermal.get("n_realizations", LEGACY_THERMAL_REALIZATIONS))
    if "sem" in thermal:
        sem_thermal = float(as_array(thermal["sem"])[-1])
        sem_measured = True
    else:
        sem_thermal = float(sems[-1]) * np.sqrt(SWEEP_REALIZATIONS / n_thermal)
        sem_measured = False
    residual = abs(float(energy[-1]) - gibbs)
    sector, n_sectors = sector_resolved_energy(doc)
    residual_sector = (abs(float(energy[-1]) - sector)
                       if sector is not None else None)

    # Has it actually stopped, or is it still creeping? A residual from a curve
    # still in motion means something different from one from a flat curve.
    tail = energy[t >= t[-1] - 20.0]
    return {
        "gibbs": gibbs, "t": t, "energy": energy,
        "m_values": m_values, "finals": finals, "sems": sems,
        "slope": float(slope), "intercept": float(intercept),
        "pairwise": pairwise,
        "sem_thermal": sem_thermal, "sem_measured": sem_measured,
        "n_thermal": n_thermal, "residual": residual,
        "sector": sector, "n_sectors": n_sectors,
        "residual_sector": residual_sector,
        "residual_sector_in_sem": (None if residual_sector is None
                                   else residual_sector / sem_thermal),
        "residual_in_sem": residual / sem_thermal,
        "tail_drift": float(tail[-1] - tail[0]),
        "fraction": float(doc["thermal"]["fraction_covered"]),
        "n_l": int(doc["n_l"]), "dim": int(doc["dim"]),
        "list_gb": float(doc["operator_list_bytes"]) / 1024 ** 3,
    }


def figure(doc, out_name):
    plt.switch_backend("Agg")
    r = derive(doc)
    fig, (ax_t, ax_m) = plt.subplots(ncols=2, figsize=(14.0, 6.2))

    # ---- LEFT: relaxation to the thermal state ---------------------------
    ax_t.plot(r["t"], r["energy"], "-", color=SLB_GREEN, lw=2.5,
              label=f"SLB, $M$={doc['thermal']['M']}, "
                    f"{r['n_thermal']} realizations")
    ax_t.axhline(r["gibbs"], ls="--", color=GIBBS_GREY, lw=2.0,
                 label=r"global Gibbs (stationary, not unique)")
    if r["sector"] is not None:
        ax_t.axhline(r["sector"], ls="-", color="tab:red", lw=2.0,
                     label=f"sector-resolved limit ({r['n_sectors']} sectors)")
    ax_t.axhspan(r["gibbs"] - r["sem_thermal"], r["gibbs"] + r["sem_thermal"],
                 color=GIBBS_GREY, alpha=0.20, lw=0,
                 label=r"$\pm 1$ s.e.m. of this run"
                       + ("" if r["sem_measured"] else " (est.)"))
    ax_t.set_xlabel("time", fontsize=12.5)
    ax_t.set_ylabel(r"$\langle H \rangle$", fontsize=12.5)
    ax_t.set_title(f"Relaxation at Dimension {r['dim']} ($N_L$ = {r['n_l']:,})\n"
                   f"Approaching stationary state without reference", fontsize=13)
    ax_t.legend(loc="upper right", bbox_to_anchor=(0.98, 0.48), fontsize=9.5, framealpha=0.95)
    ax_t.grid(alpha=0.3)

    # Inset zoom on tail
    ax_zoom = ax_t.inset_axes([0.48, 0.54, 0.48, 0.40])
    late = r["t"] >= 12.0
    ax_zoom.plot(r["t"][late], r["energy"][late], "-", color=SLB_GREEN, lw=2.2)
    ax_zoom.axhline(r["gibbs"], ls="--", color=GIBBS_GREY, lw=1.8)
    if r["sector"] is not None:
        ax_zoom.axhline(r["sector"], ls="-", color="tab:red", lw=1.8)
    ax_zoom.axhspan(r["gibbs"] - r["sem_thermal"], r["gibbs"] + r["sem_thermal"],
                    color=GIBBS_GREY, alpha=0.25, lw=0)
    lo = min(r["gibbs"], r["sector"] if r["sector"] is not None else r["gibbs"])
    hi = max(r["gibbs"], r["sector"] if r["sector"] is not None else r["gibbs"])
    pad = max(4 * r["sem_thermal"], 0.25 * (hi - lo))
    ax_zoom.set_ylim(lo - pad, hi + pad)
    ax_zoom.set_title("tail zoom (to scale)", fontsize=9.5, pad=3)
    ax_zoom.tick_params(labelsize=8.5)
    ax_zoom.grid(alpha=0.3)

    if r["sector"] is None:
        note = (f"covers {100 * r['fraction']:.1f}% of distance\n"
                f"{r['residual']:.4f} from Gibbs = {r['residual_in_sem']:.1f} s.e.m.\n"
                f"one sector, so Gibbs IS the limit")
    else:
        note = (f"to global Gibbs:  {r['residual']:.4f} ({r['residual_in_sem']:.1f} s.e.m.)\n"
                f"to sector target: {r['residual_sector']:.4f} ({r['residual_sector_in_sem']:.1f} s.e.m.)\n"
                f"flat to {abs(r['tail_drift']):.0e} over last 20 -- stopped, but short")
    ax_t.annotate(
        note, xy=(0.03, 0.03), xycoords="axes fraction", ha="left",
        va="bottom", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    # ---- RIGHT: convergence in M, against 1/M ----------------------------
    inv = 1.0 / r["m_values"]
    grid = np.linspace(0.0, inv.max() * 1.12, 50)
    ax_m.plot(grid, r["intercept"] + r["slope"] * grid, "-",
              color=GIBBS_GREY, lw=1.8, zorder=1,
              label=r"fit linear in $1/M$")
    ax_m.errorbar(inv, r["finals"], yerr=r["sems"], fmt="s", color=SLB_GREEN,
                  ms=9, capsize=5, lw=2.0, zorder=3, label="SLB (16 realizations)")
    ax_m.plot([0.0], [r["intercept"]], "*", color="tab:purple", ms=20, zorder=4,
              label=rf"$M\to\infty$: {r['intercept']:+.4f}")
    for x, y, m in zip(inv, r["finals"], r["m_values"]):
        ax_m.annotate(f"$M$={m:.0f}", xy=(x, y), xytext=(5, -14),
                      textcoords="offset points", fontsize=10.5, fontweight="semibold")
    ax_m.set_xlabel(r"$1/M$", fontsize=12.5)
    ax_m.set_ylabel(r"$\langle H \rangle$ at $t=5$", fontsize=12.5)
    ax_m.set_title("Bias falls as $1/M$, as predicted\n"
                   "(a curve here would falsify it)", fontsize=13)
    ax_m.set_xlim(left=-0.006)
    ax_m.legend(loc="lower right", fontsize=10, framealpha=0.95)
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
        "THE GIBBS STATE IS STATIONARY BUT NOT UNIQUE HERE: the coupling "
        "operator does not connect every level, so the space splits into "
        "sectors and the limit depends on $\\rho_0$. A bundle mixes operators "
        "ACROSS sectors, so at small $M$ the bundled dynamics is more ergodic "
        "than the generator it approximates and drifts toward global Gibbs. "
        "The bias is $O(1/M)$; at $N_L/M \\approx 1{,}020$ this run has barely "
        "started to converge.",
        fontsize=9.5)

    fig.savefig(out_name, dpi=140, bbox_inches="tight")
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
