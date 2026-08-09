"""
plot_accuracy_vs_M.py
=====================

ANALYSIS/FIGURE HALF of the accuracy-versus-bundle-size benchmark (Result 1).
Reads the data written by run_accuracy_vs_M.py and draws three figures per
system; runs in seconds.

  benchmark_accuracy_<s>.png     <H(t)> vs time: exact reference, SLB mean per
                                 M with a +/-1 std band.
  benchmark_coherence_<s>.png    the same for the dominant coherence <C(t)>.
  benchmark_error_decomposition_<s>.png   NEW -- the anatomy of the error at
                                 its worst moment. For each observable, t* is
                                 the time where the smallest-M estimate's
                                 RMSE(t) peaks; holding that same instant for
                                 every M, the figure plots the BIAS
                                 |mean(t*) - ref(t*)|, the SEM, the total
                                 RMSE, and the Std Dev vs M.

All derived quantities (means, bands, t*, bias, fluctuation, slopes) come from
the saved raw realizations -- nothing here re-runs any dynamics. Captions also
report the Davies-construction time separately from the propagation times, so
the two costs are never blurred.

Run:  python plot_accuracy_vs_M.py [--system ...]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from common import (
    DATA_DIR, add_settings_footer, as_array, format_slb_settings, load_data,
)

# --- CONFIGURATION (TOGGLES FOR DECOMPOSITION PLOT) ---
PLOT_RMSE = True
PLOT_BIAS = True
PLOT_SEM  = True
PLOT_STD  = True

# WHICH SYSTEM SIZE TO PLOT. run_accuracy_vs_M.py saves one file per dim
# (accuracy_vs_M_<system>_dim<D>.json); PLOT_DIM picks which to draw.
# None = auto-pick the largest dim available on disk.
PLOT_DIM = None
# COMMITTED DEFAULT: True. Result 1's error anatomy is deliberately the
# SINGLE-RUN decomposition (RMSE^2 = bias^2 + StdDev^2) -- it answers "what
# error does ONE run carry, and how does it split?", which is the question
# the fixed-sampling sweep is designed for. This is an intentional exception
# to the ensemble-headline convention of Results 2-4 (there the object being
# costed is the whole N_r-run estimate). False switches to the ensemble
# anatomy (RMSE^2 = bias^2 + SEM^2) and writes an _ensRMSE-suffixed file so
# the committed headline figure can never be silently rewritten under a
# non-default setting.
USE_SINGLE_RUN_RMSE = True
# ------------------------------------------------------

def _timing_caption(doc):
    n_real = doc["meta"]["params"]["N_REALIZATIONS"]
    t_prop = sum(row["cost"] for row in doc["slb_sweep"])
    return (f"construction vs dynamics: Davies operators built in "
            f"{doc['t_davies']*1e3:.0f} ms; reference solve "
            f"{doc['t_reference']:.1f} s; SLB propagation "
            f"{t_prop:.1f} s total ({n_real} realizations per M)")


def _size_str(name, dim):
    if name in ("spin_chain", "mixed_chain"):
        return f"{int(round(math.log2(dim)))} spins, dim {dim}"
    if name == "oscillator_bath":
        return f"Fock cutoff {dim // 2}, dim {dim}"
    return f"dim {dim}"


def _reference_label(doc):
    """Human-readable provenance for the exact reference saved with the data."""
    method = doc.get("reference_method")
    if method == "mesolve":
        return "full Lindblad reference (QuTiP mesolve)"
    prefix = "native_rk4_substeps"
    if isinstance(method, str) and method.startswith(prefix):
        substeps = method.removeprefix(prefix)
        detail = f", {substeps} substeps" if substeps else ""
        return f"full Lindblad reference (certified native RK4{detail})"
    return "full Lindblad reference"


def _curves(doc, key):
    """{M: (mean, std)} for one observable from the raw realizations."""
    out = {}
    for row in doc["slb_sweep"]:
        s = np.asarray(row[key], dtype=float)
        out[row["M"]] = (s.mean(axis=0), s.std(axis=0, ddof=1))
    return out


def accuracy_figure(plt, name, doc, tlist, reference, curves,
                    obs_math, fname_suffix, subtitle):
    """Single panel: observable vs time, one SLB curve per M."""
    meta = doc["meta"]["params"]
    reference_label = _reference_label(doc)
    
    # ---------------------------------------------------------
    # EDIT THIS LIST TO QUICKLY CHANGE WHICH M VALUES ARE SHOWN
    # Set to None if you want to plot all available M values.
    # ---------------------------------------------------------
    M_TO_PLOT = {
        "spin_chain": [8, 16, 32, 64],
        "mixed_chain": [8, 16, 32, 64],
        "oscillator_bath": [2, 8]
    }
    system_m_list = M_TO_PLOT.get(name)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.5))
    
    ax.plot(tlist, reference, "k-", lw=1.4, alpha=0.7,
            label=reference_label, zorder=1)
    
    palette = ["tab:orange", "tab:blue", "tab:green", "tab:purple",
               "tab:red", "tab:brown", "tab:pink", "tab:olive"]
               
    for (m_eff, (mean, std)), col in zip(sorted(curves.items()), palette):
        # Filter based on the specific system's list
        if system_m_list is not None and m_eff not in system_m_list:
            continue
            
        ax.plot(tlist, mean, "-", color=col, lw=1.6, label=f"SLB, M={m_eff}",
                zorder=3)
        ax.fill_between(tlist, mean - std, mean + std, color=col, alpha=0.16)
        
    ax.set_xlabel("time")
    ax.set_ylabel(rf"${obs_math}$")
    ax.set_title(rf"{name} ({_size_str(name, doc['dim'])}, "
                 rf"$N_L$={doc['n_l']}): {subtitle}")
    ax.legend(frameon=False)
    fig.tight_layout()
    
    # Update footer to accurately reflect only the M values currently plotted
    plotted_m = sorted(curves) if not system_m_list else sorted(m for m in curves if m in system_m_list)
    
    add_settings_footer(
        fig,
        format_slb_settings(M=plotted_m, substeps=doc["meta"]["substeps"],
                            n_realizations=meta["N_REALIZATIONS"]),
        f"shaded band = \u00b11 std over realizations; {reference_label}",
        _timing_caption(doc),
    )
    
    fig.savefig(f"benchmark_{fname_suffix}_{name}.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benchmark_{fname_suffix}_{name}.png")


def peak_error_anatomy(doc, key, ref):
    """(t_star_index, m_values, bias(t*) per M, fluctuation(t*) per M)."""
    rows = sorted(doc["slb_sweep"], key=lambda r: r["M"])
    n = np.asarray(rows[0][key], dtype=float).shape[0]
    s0 = np.asarray(rows[0][key], dtype=float)
    bias0 = np.abs(s0.mean(axis=0) - ref)
    fluct0 = s0.std(axis=0, ddof=1)
    sem0 = fluct0 / np.sqrt(n)
    
    if USE_SINGLE_RUN_RMSE:
        rmse0 = np.sqrt(bias0 ** 2 + fluct0 ** 2)
    else:
        rmse0 = np.sqrt(bias0 ** 2 + sem0 ** 2)
        
    t_star = int(np.argmax(rmse0))
    
    m_values, bias, fluct = [], [], []
    for row in rows:
        s = np.asarray(row[key], dtype=float)
        m_values.append(row["M"])
        bias.append(abs(float(s.mean(axis=0)[t_star]) - float(ref[t_star])))
        fluct.append(float(s.std(axis=0, ddof=1)[t_star]))
    return t_star, np.array(m_values), np.array(bias), np.array(fluct)


def _fit_label(base, m, y, floor=None):
    """Fitted power-law label. If `floor` (the SEM) is given, points at or below
    it are sampling noise, not signal: fitting them would advertise a bogus
    convergence rate, so we report an upper bound instead of a slope."""
    if floor is not None:
        # "Resolved" means clearly ABOVE the sampling floor, not marginally so:
        # a point that pokes over the SEM by a hair is still mostly noise, and
        # fitting two such points advertises a rate that is pure wobble. Require
        # a factor-2 margin and at least three points before quoting a slope.
        clear = np.asarray(y) > 2.0 * np.asarray(floor)
        if int(clear.sum()) < 3:
            return base + r"  — below noise floor (upper bound)"
        m, y = np.asarray(m)[clear], np.asarray(y)[clear]
    return _fit_label_raw(base, m, y)


def _fit_label_raw(base, m, y):
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return base
    e = float(np.polyfit(np.log(m[ok]), np.log(y[ok]), 1)[0])
    return base + rf"  — $\propto M^{{{e:.2f}}}$"


def decomposition_figure(plt, name, doc, tlist):
    """Bias, SEM, RMSE, and Std Dev vs M at the peak-error instant, per observable."""
    meta = doc["meta"]["params"]
    n_real = meta["N_REALIZATIONS"]
    ia, ib = doc["coherence_pair"]
    panels = [
        ("samples_energy", as_array(doc["reference_energy"]),
         r"\langle H\rangle", "energy"),
        ("samples_coherence", as_array(doc["reference_coherence"]),
         rf"\langle C_{{{ia},{ib}}}\rangle", "coherence"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    t_stars = []
    for ax, (key, ref, obs_math, obs_name) in zip(axes, panels):
        t_star, m, bias, fluct = peak_error_anatomy(doc, key, ref)
        t_stars.append(tlist[t_star])
        
        # Calculate SEM and requested RMSE
        sem = fluct / np.sqrt(n_real)
        
        if USE_SINGLE_RUN_RMSE:
            rmse = np.sqrt(bias**2 + fluct**2)
            rmse_label_prefix = "single-run RMSE"
            footer_math = "RMSE\u00b2 = bias\u00b2 + StdDev\u00b2; expected: bias \u221d 1/M, StdDev \u221d 1/\u221aM"
        else:
            rmse = np.sqrt(bias**2 + sem**2)
            rmse_label_prefix = "total RMSE (ensemble)"
            footer_math = "RMSE\u00b2 = bias\u00b2 + SEM\u00b2; expected: bias \u221d 1/M, SEM \u221d 1/\u221aM"
        
        # Conditionally plot based on toggles
        if PLOT_RMSE:
            ax.loglog(m, rmse, "v-", color="black", lw=2.0, ms=6, zorder=4,
                      label=_fit_label(rmse_label_prefix, m, rmse))
        if PLOT_BIAS:
            ax.loglog(m, bias, "o-", color="tab:red", lw=1.8, ms=6, zorder=3,
                      label=_fit_label("bias  $|{\\rm mean}-{\\rm ref}|$", m, bias,
                                       floor=sem))
        if PLOT_SEM:
            ax.loglog(m, sem, "s-", color="tab:blue", lw=1.8, ms=6, zorder=2,
                      label=_fit_label("SEM  (fluctuation/$\\sqrt{N}$)", m, sem))
        if PLOT_STD:
            ax.loglog(m, fluct, "d-", color="tab:green", lw=1.8, ms=6, zorder=1,
                      label=_fit_label("Std Dev (fluctuation)", m, fluct))
                  
        ax.set_xlabel("bundle size $M$")
        ax.set_title(rf"${obs_math}$ at $t^*={tlist[t_star]:.2f}$"
                     rf"  ({obs_name} peak error)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        
    axes[0].set_ylabel("error at $t^*$")
    fig.suptitle(rf"{name} ({_size_str(name, doc['dim'])}, $N_L$={doc['n_l']}): "
                 rf"error anatomy at the worst moment, fixed sampling")
    fig.tight_layout()
    
    add_settings_footer(
        fig,
        f"t* = argmax of RMSE(t) of the smallest-M estimate, held for all M; "
        f"{n_real} realizations at every M",
        f"{footer_math}; {doc['meta']['substeps']} RK4 substep(s)/step",
        _timing_caption(doc),
    )
    
    suffix = "" if USE_SINGLE_RUN_RMSE else "_ensRMSE"
    fig.savefig(f"benchmark_error_decomposition_{name}{suffix}.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benchmark_error_decomposition_{name}{suffix}.png  "
          f"(t* energy={t_stars[0]:.2f}, coherence={t_stars[1]:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", default="all",
                    choices=["spin_chain", "mixed_chain", "oscillator_bath", "all"])
    args = ap.parse_args()
    names = (["spin_chain", "mixed_chain", "oscillator_bath"] if args.system == "all"
             else [args.system])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import re

    def _resolve(name):
        if PLOT_DIM is not None:
            return f"accuracy_vs_M_{name}_dim{PLOT_DIM}.json"
        paths = list(DATA_DIR.glob(f"accuracy_vs_M_{name}_dim*.json"))
        if not paths:
            return f"accuracy_vs_M_{name}.json"
        best = max(
            paths,
            key=lambda path: int(re.search(r"_dim(\d+)", path.name).group(1)),
        )
        return best.name

    for name in names:
        fname = _resolve(name)
        doc = load_data(fname)
        print(f"[{name}] plotting {fname} (dim {doc.get('dim','?')})")
        t = doc["meta"]["tlist"]
        tlist = np.linspace(t["t0"], t["t1"], t["n"])
        ia, ib = doc["coherence_pair"]
        print(f"[{name}]")
        accuracy_figure(
            plt, name, doc, tlist, as_array(doc["reference_energy"]),
            _curves(doc, "samples_energy"),
            obs_math=r"\langle H\rangle", fname_suffix="accuracy",
            subtitle="SLB vs full Lindblad reference (energy)",
        )
        accuracy_figure(
            plt, name, doc, tlist, as_array(doc["reference_coherence"]),
            _curves(doc, "samples_coherence"),
            obs_math=rf"\langle C_{{{ia},{ib}}}\rangle",
            fname_suffix="coherence",
            subtitle=rf"SLB vs reference (coherence on eigenstates {ia},{ib})",
        )
        decomposition_figure(plt, name, doc, tlist)


if __name__ == "__main__":
    main()
