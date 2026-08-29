"""Merge a re-timed cost_scaling file over its committed counterpart, safely.

Re-running a benchmark to fix its wall-clocks is only valid if nothing else
changed. Twice this week a job was submitted whose settings silently differed
from the run it was replacing -- once with the wrong substep count, once with
the wrong reference cap -- and neither failed. Both produced a plausible file
full of numbers measuring something other than what was asked for.

So this refuses to merge unless the incoming file agrees with the committed one
on everything that is not a timing:

  * the same system, package version and degeneracy tolerance
  * the same substep count and reference cap -- the two that went wrong
  * the same dimensions, with the same operator counts at each
  * the same RMSE at every bundle size, to a tight tolerance

That last check is the important one. SLB is deterministic at a fixed seed, so a
re-run that changes only the clock must reproduce every accuracy number
exactly. If an RMSE moves, something other than timing changed and the merge is
refused. Verified in practice: the mixed chain's re-run reproduced all eight
RMSE values to every printed digit while its reference time fell 27x.

Usage:

    python merge_retimed.py <retimed.json>            # report only
    python merge_retimed.py <retimed.json> --apply    # write it over data/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
RMSE_TOL = 1e-9        # deterministic at fixed seed; this is a formatting margin


def _points(doc):
    return {p["dim"]: p for p in doc["points"]}


def _sweep(point):
    return {e["M"]: e["rmse"] for e in point.get("m_sweep", [])}


def check(incoming: dict, committed: dict) -> list[str]:
    """Everything that is not a wall-clock must be identical."""
    problems: list[str] = []
    im, cm = incoming["meta"], committed["meta"]

    for label, got, want in (
        ("system", im["params"].get("system"), cm["params"].get("system")),
        ("qutip_bundling", im.get("qutip_bundling"), cm.get("qutip_bundling")),
        ("degeneracy_tol", im["davies"].get("degeneracy_tol"),
         cm["davies"].get("degeneracy_tol")),
        ("substeps", im.get("substeps"), cm.get("substeps")),
        ("native_ref_max_dim", im["params"].get("native_ref_max_dim"),
         cm["params"].get("native_ref_max_dim")),
    ):
        if got != want:
            problems.append(f"{label}: re-run has {got!r}, committed has {want!r}")

    ip, cp = _points(incoming), _points(committed)
    missing = sorted(set(cp) - set(ip))
    if missing:
        problems.append(f"dimensions present in the committed file but not the "
                        f"re-run: {missing}")

    for dim in sorted(set(ip) & set(cp)):
        if ip[dim].get("n_l") != cp[dim].get("n_l"):
            problems.append(f"dim {dim}: N_L {ip[dim].get('n_l')} against "
                            f"committed {cp[dim].get('n_l')}")
        a, b = _sweep(ip[dim]), _sweep(cp[dim])
        for m in sorted(set(a) & set(b)):
            if abs(a[m] - b[m]) > RMSE_TOL * max(1.0, abs(b[m])):
                problems.append(
                    f"dim {dim}, M={m}: RMSE moved, {a[m]:.9e} against "
                    f"committed {b[m]:.9e} -- a re-timing must not change this")
    return problems


def report(incoming: dict, committed: dict) -> None:
    ip, cp = _points(incoming), _points(committed)
    print(f"{'dim':>5s} {'N_L':>6s} {'ref before':>12s} {'ref after':>11s} "
          f"{'x':>6s} {'ratio before':>13s} {'ratio after':>12s} {'spread':>8s}")
    for dim in sorted(set(ip) & set(cp)):
        i, c = ip[dim], cp[dim]
        ri, rc = i.get("t_native_ref"), c.get("t_native_ref")
        si, sc = i.get("t_slb_fixed"), c.get("t_slb_fixed")
        if not (ri and rc and si and sc):
            continue
        reps = i.get("t_native_ref_repeats") or [ri]
        spread = max(reps) / min(reps) if min(reps) > 0 else float("nan")
        print(f"{dim:5d} {i.get('n_l', 0):6d} {rc:12.1f} {ri:11.1f} {rc/ri:6.1f} "
              f"{rc/sc:12.1f}x {ri/si:11.1f}x {spread:7.2f}x")
        if len(reps) < 2:
            print(f"      (dim {dim} carries a single timing sample; "
                  f"--timing-repeats gives it a spread)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("retimed", type=Path, help="the re-run's JSON")
    ap.add_argument("--apply", action="store_true",
                    help="write it over the committed file (a .bak is kept)")
    args = ap.parse_args()

    incoming = json.loads(args.retimed.read_text(encoding="utf-8"))
    system = incoming["meta"]["params"].get("system")
    target = DATA_DIR / f"cost_scaling_{system}.json"
    if not target.exists():
        print(f"no committed file at {target}")
        return 2
    committed = json.loads(target.read_text(encoding="utf-8"))

    print(f"re-run   {args.retimed}")
    print(f"against  {target}\n")
    report(incoming, committed)

    problems = check(incoming, committed)
    if problems:
        print("\nREFUSING TO MERGE — the re-run is not measuring the same thing:")
        for p in problems:
            print(f"  * {p}")
        print("\nFix the run settings and re-submit, or if the difference is "
              "deliberate, say so in the commit rather than merging silently.")
        return 1

    print("\nevery non-timing quantity matches: settings, dimensions, "
          "operator counts, and every RMSE")
    if not args.apply:
        print("(report only; pass --apply to write)")
        return 0

    shutil.copy2(target, target.with_suffix(".json.bak"))
    target.write_text(json.dumps(incoming, indent=2), encoding="utf-8")
    print(f"merged. previous file kept at {target.with_suffix('.json.bak')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
