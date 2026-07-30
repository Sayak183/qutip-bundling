"""Plot the certified native-RK4 spin-chain references at high dimension.

The source JSON files contain the energy trajectory and the full-state
substep-convergence check.  Their companion NPZ archives are not needed for
this figure.

Example
-------
    python benchmarks/plot_high_dim_spin_reference.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

from common import DATA_DIR, load_data


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).with_name(
    "benchmark_high_dim_spin_reference.png"
)


def discover_dims() -> list[int]:
    """Every certified dimension present in data/, ascending.

    Defaulting to the files on disk rather than a hard-coded pair keeps the
    committed figure in step with the committed data: adding a dimension and
    re-running with no arguments redraws all of them, instead of silently
    reverting the figure to the two dimensions that happened to exist when
    this script was written.
    """
    dims = []
    for path in DATA_DIR.glob("high_dim_reference_spin_chain_dim*.json"):
        match = re.search(r"dim(\d+)\.json$", path.name)
        if match:
            dims.append(int(match.group(1)))
    return sorted(dims)


def _load_reference(dim: int) -> dict:
    filename = f"high_dim_reference_spin_chain_dim{dim}.json"
    document = load_data(filename)
    point = document["point"]
    selfcheck = point["selfcheck"]
    if point["dim"] != dim:
        raise ValueError(
            f"{filename} records dimension {point['dim']}, expected {dim}"
        )
    if not selfcheck.get("passed", False):
        raise ValueError(f"{filename} is not a certified reference")
    if "trace_distance_by_time" not in selfcheck:
        raise ValueError(
            f"{filename} predates the per-time certification arrays "
            f"(qutip_bundling {document['meta'].get('qutip_bundling', '?')}); "
            f"it records only a scalar summary, so the convergence panel "
            f"cannot be drawn. Regenerate it with "
            f"run_high_dim_spin_reference.py --dim {dim}."
        )
    return document


def _time_grid(document: dict) -> np.ndarray:
    spec = document["meta"]["tlist"]
    return np.linspace(spec["t0"], spec["t1"], spec["n"])


def figure(documents: list[dict]):
    fig, (ax_energy, ax_check) = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        constrained_layout=True,
    )

    # One colour per reference, generated from the count. A fixed list silently
    # truncated the figure via zip() as soon as more dimensions were passed
    # than it had entries, dropping curves with no error.
    base = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    if len(documents) <= len(base):
        colors = base[:len(documents)]
    else:
        cmap = matplotlib.colormaps["viridis"]
        colors = [cmap(i / max(len(documents) - 1, 1))
                  for i in range(len(documents))]

    for color, document in zip(colors, documents):
        point = document["point"]
        dim = point["dim"]
        sites = point["n_sites"]
        times = _time_grid(document)
        energy_density = np.asarray(
            point["reference_energy"], dtype=float
        ) / sites
        trace_distance = np.asarray(
            point["selfcheck"]["trace_distance_by_time"], dtype=float
        )
        label = (
            rf"$d={dim}$ ({sites} spins, $N_L={point['n_l']}$)"
        )

        ax_energy.plot(
            times,
            energy_density,
            color=color,
            linewidth=2,
            label=label,
        )
        positive = trace_distance > 0
        ax_check.semilogy(
            times[positive],
            trace_distance[positive],
            color=color,
            linewidth=2,
        )

    tolerance = min(
        document["point"]["selfcheck"]["tol"] for document in documents
    )
    ax_check.axhline(
        tolerance,
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label=rf"certification tolerance $={tolerance:g}$",
    )

    ax_energy.set_title("Native-RK4 energy density")
    ax_energy.set_xlabel("time")
    ax_energy.set_ylabel(r"total energy / number of spins")
    ax_energy.grid(True, alpha=0.25)

    # Both panels share one dimension-to-colour mapping, so a single legend
    # outside the axes serves both. Repeating it per panel obscured curves in
    # the left panel and printed the same substep pair once per dimension in
    # the right one. It is attached to the figure, not an axes, so it sits
    # clear of the plots instead of between them.

    # The substep pair is identical across a sweep in practice; say it once in
    # the title, and only fall back to naming it per curve if a mixed set is
    # ever passed. Either way it is read from the data, never hard-coded.
    pairs = {tuple(d["point"]["selfcheck"]["substeps_pair"]) for d in documents}
    if len(pairs) == 1:
        low, high = next(iter(pairs))
        check_title = (f"Full-state convergence certification\n"
                       f"(substeps {low} vs {high})")
    else:
        check_title = "Full-state convergence certification (mixed substeps)"
    ax_check.set_title(check_title)
    ax_check.set_xlabel("time")
    ax_check.set_ylabel("trace distance")
    ax_check.grid(True, which="both", alpha=0.25)
    ax_check.legend(fontsize=9, loc="lower left")

    handles, labels = ax_energy.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside right center", fontsize=9,
        frameon=False, title="reference",
    )

    fig.suptitle(
        "Spin-chain references beyond the practical mesolve wall",
        fontsize=14,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=None,
        help="certified reference dimensions to plot "
             "(default: every one found in data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output PNG path",
    )
    args = parser.parse_args()

    dims = args.dims if args.dims else discover_dims()
    if not dims:
        raise SystemExit(
            f"no certified reference JSON found in {DATA_DIR}"
        )
    documents = [_load_reference(dim) for dim in dims]
    fig = figure(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    sources = ", ".join(
        str(
            DATA_DIR
            / f"high_dim_reference_spin_chain_dim{dim}.json"
        )
        for dim in dims
    )
    print(f"wrote {args.output}")
    print(f"sources: {sources}")


if __name__ == "__main__":
    main()
