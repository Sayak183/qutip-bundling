"""
plot_method_comparison.py
=========================

ANALYSIS HALF of the four-method comparison. Reads the JSON written by
run_method_comparison.py and draws, per observable:

  * accuracy versus cost -- the figure the comparison exists for. Each method
    is a point or a curve in the (wall-clock, error) plane, so "which method
    gets me this accuracy for the least compute" is read off directly.
    SLB traces a curve as M grows, mcsolve is a single fixed-budget point,
    and the exact solvers sit at their own cost with error at the integrator
    floor.
  * the dynamics themselves, so the curves can be eyeballed against the
    certified reference.

The accuracy target is applied HERE, not in the run script: the runner saves
raw per-realization samples, so any target can be evaluated after the fact
without recomputing. That is the same run/plot split as the other Results.

Error is the time-averaged RMSE from common.tavg_rmse -- for SLB it combines
the bias of the ensemble mean with its standard error, so a method is not
rewarded for being merely noisy about the right answer.

Wall-clock numbers are only comparable within one job on one node. The runner
records the Slurm job and hostname; this script refuses to draw a cost axis
across files that disagree, rather than silently producing a meaningless
figure.

Run:  python plot_method_comparison.py [--system SYSTEM] [--dims ...]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

from common import DATA_DIR, load_data, tavg_rmse

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SYSTEMS = ("spin_chain", "mixed_chain", "oscillator_bath")
METHOD_STYLE = {
    "native":  dict(color="#333333", marker="s", label="native RK4 (full dissipator)"),
    "mesolve": dict(color="#0072B2", marker="D", label="qutip mesolve (exact)"),
    "mcsolve": dict(color="#D55E00", marker="^", label="qutip mcsolve"),
    "slb":     dict(color="#009E73", marker="o", label="SLB (this package)"),
    "jackknife": dict(color="#CC79A7", marker="v",
                      label="SLB + jackknife-2 (bias corrected)"),
}


def discover_dims(system: str) -> list[int]:
    dims = []
    for path in DATA_DIR.glob(f"method_comparison_{system}_dim*.json"):
        match = re.search(r"dim(\d+)\.json$", path.name)
        if match:
            dims.append(int(match.group(1)))
    return sorted(dims)


def load_point(system: str, dim: int) -> dict:
    return load_data(f"method_comparison_{system}_dim{dim}.json")


def execution_key(document: dict):
    """(hostname, slurm job) identifying the allocation that produced a file."""
    execution = document["meta"].get("execution") or {}
    slurm = execution.get("slurm") or {}
    return (execution.get("hostname"), slurm.get("job_id"))


def mean_curve(values) -> np.ndarray:
    """A single curve from a saved observable.

    Tolerates the per-trajectory shape ``(ntraj, n_times)``: files written
    before the mcsolve averaging fix stored every trajectory instead of their
    mean, and the mean is recoverable from them, so those runs stay usable
    rather than needing to be repeated.
    """
    array = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    if array.ndim == 2:
        return array.mean(axis=0)
    return array


def method_errors(point: dict, observable: str):
    """[(method, wall_seconds, error, annotation, n_samples), ...].

    SLB contributes one entry per bundle size; every other method contributes
    at most one.

    ``n_samples`` is how many independent samples the wall-clock covers: 500
    trajectories for `mcsolve`, ``n_runs`` realizations for SLB, and 1 for the
    deterministic solvers. It is what makes the parallel-limit axis possible --
    both stochastic methods are embarrassingly parallel over their samples, so
    ``wall_s / n_samples`` is the time each would take given one core per
    sample, while ``wall_s`` is the time on a single core. The honest speedup
    lies between the two and depends on the core count available.
    """
    index = point["observables"].index(observable)
    reference = mean_curve(point["reference"]["curves"][observable])
    rows = []

    for name in ("native", "mesolve", "mcsolve"):
        entry = point["methods"].get(name)
        if not entry or "skipped" in entry:
            continue
        curve = mean_curve(entry["curves"][observable])
        # A single deterministic curve: no ensemble, so the RMSE reduces to the
        # time-averaged absolute deviation from the reference.
        error = float(np.mean(np.abs(curve - reference)))
        samples = int(entry.get("ntraj", 1))
        note = f"ntraj={samples}" if name == "mcsolve" else None
        rows.append((name, float(entry["wall_s"]), error, note, samples))

    for family in ("slb", "jackknife"):
        for row in point["methods"].get(family, []):
            if row.get("diverged"):
                continue
            samples = np.asarray(row["samples"], dtype=float)[:, index, :]
            rows.append((family, float(row["wall_s"]),
                         tavg_rmse(samples, reference), f"M={row['M']}",
                         int(row.get("n_runs", samples.shape[0]))))
    return rows


def bias_comparison(point, observable):
    """[(M, uncorrected |bias|, jackknife |bias|), ...] for one observable.

    Both are measured against the same reference from the same draws, so the
    difference isolates what the correction actually removes. Returns [] when
    the jackknife was not run.
    """
    index = point["observables"].index(observable)
    reference = mean_curve(point["reference"]["curves"][observable])
    out = []
    for row in point["methods"].get("jackknife", []):
        if row.get("diverged"):
            continue
        corrected = np.asarray(row["samples"], dtype=float)[:, index, :]
        direct = np.asarray(row["direct_samples"], dtype=float)[:, index, :]
        out.append((
            row["M"],
            float(np.mean(np.abs(direct.mean(axis=0) - reference))),
            float(np.mean(np.abs(corrected.mean(axis=0) - reference))),
        ))
    return sorted(out)


EXACT_METHODS = ("native", "mesolve")


def _draw_cost_panel(ax, points, observable, per_sample: bool):
    """One accuracy-versus-cost panel.

    ``per_sample`` divides each wall-clock by the number of independent samples
    it covers, which is the limit of one core per sample. The deterministic
    solvers have one sample, so they do not move between the two panels -- only
    the stochastic methods do, and that difference IS the parallel headroom.
    """
    dims = sorted(points)
    alphas = np.linspace(0.45, 1.0, len(dims))
    seen = set()

    def cost(wall, samples):
        return wall / samples if per_sample else wall

    for alpha, dim in zip(alphas, dims):
        rows = method_errors(points[dim], observable)

        for family in ("slb", "jackknife"):
            curve = sorted([r for r in rows if r[0] == family],
                           key=lambda r: cost(r[1], r[4]))
            if not curve:
                continue
            style = METHOD_STYLE[family]
            ax.plot([cost(r[1], r[4]) for r in curve], [r[2] for r in curve],
                    color=style["color"], marker=style["marker"],
                    alpha=alpha, linewidth=1.6, markersize=5,
                    label=style["label"] if family not in seen else None)
            seen.add(family)
            if family == "slb":
                ax.annotate(f"d={dim}", (cost(curve[0][1], curve[0][4]), curve[0][2]),
                            textcoords="offset points", xytext=(-4, 6),
                            fontsize=8, color=style["color"], alpha=alpha)

        for name, wall, error, _note, samples in rows:
            if name in ("slb", "jackknife"):
                continue
            style = METHOD_STYLE[name]
            x = cost(wall, samples)
            # Every method carries its dimension. Without this the exact
            # solvers are unlabelled points sitting four decades below SLB,
            # which reads as SLB being the worst method rather than as the
            # only approximate one on that part of the axis.
            ax.plot([x], [error], color=style["color"], marker=style["marker"],
                    alpha=alpha, markersize=8, linestyle="none",
                    markerfacecolor="none" if name in EXACT_METHODS else style["color"],
                    markeredgewidth=1.8,
                    label=style["label"] if name not in seen else None)
            ax.annotate(f"d={dim}", (x, error), textcoords="offset points",
                        xytext=(5, -10), fontsize=7, color=style["color"],
                        alpha=alpha)
            seen.add(name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("seconds per sample (one core per sample)" if per_sample
                  else "wall-clock seconds, single core")
    ax.grid(True, which="both", alpha=0.25)
    return seen


def figure_accuracy_vs_cost(system, points, observable):
    """Two panels: the single-core total, and the one-core-per-sample limit.

    Both stochastic methods parallelize trivially over their samples, so a
    single serial number overstates the gap between them by exactly the ratio
    of their sample counts. Drawing both ends makes the range explicit instead
    of quietly picking the flattering one.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True,
                             sharey=True)
    _draw_cost_panel(axes[0], points, observable, per_sample=False)
    _draw_cost_panel(axes[1], points, observable, per_sample=True)

    axes[0].set_ylabel(f"time-averaged error in {observable}")
    axes[0].set_title("single core: total wall-clock", fontsize=10)
    axes[1].set_title("one core per sample: wall-clock / samples", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=8, loc="best")
    fig.suptitle(
        f"{system}: accuracy versus cost -- {observable}\n"
        f"lower-left is better; shade darkens with dimension; "
        f"open markers are exact solvers (error = integrator floor)",
        fontsize=12)
    return fig


def figure_dynamics(system, points, observable):
    dims = sorted(points)
    fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.8),
                             constrained_layout=True, squeeze=False)
    for ax, dim in zip(axes[0], dims):
        point = points[dim]
        index = point["observables"].index(observable)
        grid = point["reference"]["curves"][observable]
        times = np.linspace(0.0, 1.0, len(grid))
        spec = point["meta"]["tlist"] if "meta" in point else None
        if spec:
            times = np.linspace(spec["t0"], spec["t1"], spec["n"])
        reference = mean_curve(grid)
        ax.plot(times, reference, color="#333333", linewidth=2.4,
                label="certified reference")
        for name in ("mesolve", "mcsolve"):
            entry = point["methods"].get(name)
            if not entry or "skipped" in entry:
                continue
            ax.plot(times, mean_curve(entry["curves"][observable]),
                    color=METHOD_STYLE[name]["color"], linewidth=1.3,
                    linestyle="--", label=METHOD_STYLE[name]["label"])
        slb = [r for r in point["methods"].get("slb", []) if not r.get("diverged")]
        if slb:
            best = max(slb, key=lambda r: r["M"])
            mean = np.asarray(best["samples"], dtype=float)[:, index, :].mean(axis=0)
            ax.plot(times, mean, color=METHOD_STYLE["slb"]["color"],
                    linewidth=1.3, linestyle=":",
                    label=f"SLB M={best['M']}")
        ax.set_title(f"d={dim}  (N_L={point['n_l']})")
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel(observable)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"{system}: {observable} against the certified reference",
                 fontsize=13)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--system", choices=SYSTEMS, default=None,
                        help="default: every system with data present")
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="default: every dimension found")
    parser.add_argument("--observables", nargs="+", default=None,
                        help="default: every observable in the files")
    parser.add_argument("--allow-mixed-jobs", action="store_true",
                        help="draw cost axes even when the files come from "
                             "different jobs or hosts (their wall-clock times "
                             "are then not comparable)")
    args = parser.parse_args()

    systems = [args.system] if args.system else [
        s for s in SYSTEMS if discover_dims(s)]
    if not systems:
        raise SystemExit(f"no method_comparison_*.json found in {DATA_DIR}")

    for system in systems:
        dims = args.dims if args.dims else discover_dims(system)
        documents = {d: load_point(system, d) for d in dims}
        points = {d: doc["point"] for d, doc in documents.items()}
        for d, doc in documents.items():
            points[d]["meta"] = doc["meta"]

        keys = {execution_key(doc) for doc in documents.values()}
        if len(keys) > 1 and not args.allow_mixed_jobs:
            listed = "\n".join(f"  d={d}: host={execution_key(doc)[0]}, "
                               f"job={execution_key(doc)[1]}"
                               for d, doc in sorted(documents.items()))
            raise SystemExit(
                f"{system}: these files come from different allocations, so "
                f"their wall-clock times cannot be compared:\n{listed}\n"
                f"Re-run them in one job, or pass --allow-mixed-jobs to draw "
                f"the figure anyway."
            )

        observables = args.observables or points[dims[0]]["observables"]
        for observable in observables:
            fig = figure_accuracy_vs_cost(system, points, observable)
            out = Path(__file__).with_name(
                f"benchmark_comparison_{system}_{observable}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out.name}")

        fig = figure_dynamics(system, points, observables[0])
        out = Path(__file__).with_name(
            f"benchmark_comparison_dynamics_{system}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
