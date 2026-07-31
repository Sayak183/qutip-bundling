"""
compact_comparison_data.py
==========================

Shrink method_comparison_*.json files in place, without changing what they
mean.

Two things inflate them:

  * mcsolve curves saved unaveraged. MC_OPTIONS sets keep_runs_results, so
    qutip returns per-trajectory arrays of shape (ntraj, n_times); a runner
    that stored those directly kept 500 trajectories where one mean curve and
    its spread were wanted. The mean is exactly recoverable, so those runs do
    not need repeating -- they need averaging.
  * indented JSON. These files carry raw sample arrays, and one number per
    line costs several bytes of whitespace per value.

Averaging is only applied where a curve is two-dimensional, so running this on
already-correct files rewrites them compactly and changes no number.

Run:  python compact_comparison_data.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import DATA_DIR, save_data


PATTERN = "method_comparison_*.json"


def average_mcsolve(point: dict) -> int:
    """Collapse per-trajectory mcsolve curves to their mean. Returns #changed."""
    entry = point.get("methods", {}).get("mcsolve")
    if not entry or "curves" not in entry:
        return 0
    changed = 0
    for label, values in list(entry["curves"].items()):
        array = np.asarray(values, dtype=float)
        if array.ndim == 2:
            entry["curves"][label] = array.mean(axis=0).tolist()
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    paths = sorted(DATA_DIR.glob(PATTERN))
    if not paths:
        raise SystemExit(f"no {PATTERN} found in {DATA_DIR}")

    total_before = total_after = 0
    for path in paths:
        before = path.stat().st_size
        total_before += before
        document = json.loads(path.read_text(encoding="utf-8"))
        averaged = average_mcsolve(document["point"])

        if args.dry_run:
            note = f"{averaged} mcsolve curves would be averaged" if averaged \
                else "already averaged"
            print(f"{path.name}: {before/1e6:.2f} MB, {note}")
            continue

        meta = document.pop("meta")
        save_data(path.name, meta, compact=True, **document)
        after = path.stat().st_size
        total_after += after
        note = f"averaged {averaged} mcsolve curves; " if averaged else ""
        print(f"{path.name}: {note}{before/1e6:.2f} -> {after/1e6:.2f} MB")

    if not args.dry_run:
        print(f"\ntotal {total_before/1e6:.1f} -> {total_after/1e6:.1f} MB "
              f"({100*(1-total_after/max(total_before,1)):.0f}% smaller)")


if __name__ == "__main__":
    main()
