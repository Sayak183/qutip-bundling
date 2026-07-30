"""Shared safety controls for benchmark data-generation commands."""

from __future__ import annotations

import sys
from pathlib import Path


def add_safety_arguments(parser, systems):
    """Require explicit scope and add non-interactive overwrite protection."""
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--system",
        choices=tuple(systems),
        help="run one explicitly selected system",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="explicitly run every configured system",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of existing benchmark JSON outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned work and output files without running solvers",
    )


def add_max_full_dim_argument(parser, default):
    """Expose the exact-solver dimension cap as an explicit opt-in.

    ``common.MAX_FULL_DIM`` is a guard for the reference laptop: past it, one
    ``mesolve`` call can run for hours and the elapsed-time budget cannot
    intervene once the call has started. On a machine that can afford the
    solve, the cap has to be liftable, otherwise the run silently records
    ``t_full = NaN`` at the very dimensions the exact reference is wanted for.

    The default preserves the module value, so nothing changes unless the flag
    is passed. The effective value is stamped into the run metadata.
    """
    parser.add_argument(
        "--max-full-dim", type=int, default=default,
        help=f"largest Hilbert dimension at which the exact mesolve reference "
             f"is attempted (default {default}). Raise only on a machine "
             f"sized for it: the spin chain at dim 64 took ~11,160 s on the "
             f"reference laptop, and the call cannot be interrupted by "
             f"--full-budget once started.",
    )


def selected_systems(args, systems):
    """Resolve the required --system/--all scope."""
    return list(systems) if args.all else [args.system]


def preflight_run(plans, *, overwrite, dry_run):
    """Print a plan and refuse accidental replacement before solver work.

    ``plans`` is an iterable of ``(description, output_path)`` pairs. Returns
    ``True`` when computation should proceed and ``False`` for a dry run.
    """
    plans = [(description, Path(path).resolve()) for description, path in plans]
    if not plans:
        raise ValueError("benchmark plan is empty")

    print("Benchmark plan:")
    existing = []
    for description, path in plans:
        exists = path.exists()
        print(f"  - {description}")
        print(f"    output: {path} [{'exists' if exists else 'new'}]")
        if exists:
            existing.append(path)

    if dry_run:
        if existing and not overwrite:
            print("Dry run note: existing outputs would require --overwrite.")
        print("Dry run only: no solvers ran and no files were written.")
        return False

    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        sys.stdout.flush()
        raise SystemExit(
            "Refusing to replace existing benchmark data:\n"
            f"{formatted}\n"
            "Review the plan with --dry-run, then pass --overwrite only if "
            "replacement is intentional."
        )

    if existing:
        print("Overwrite enabled for the existing outputs listed above.")
    return True
