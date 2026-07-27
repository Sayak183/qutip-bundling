"""
export_csv.py
=============

Flatten the canonical benchmark data files (top-level data/*.json, written by
the run_*.py scripts) into Excel-friendly CSVs under data/csv/. The JSON stays
the machine-readable source of truth, with full metadata; the CSVs are the
human/collaborator view of the same numbers. Superseded files under
data/legacy/ are deliberately excluded from the default export.

Per data file it writes up to two CSVs:

  <stem>_dynamics.csv   The observable dynamics over time, in tidy long
                        format: one row per (dimension, observable, series,
                        time) with the ensemble MEAN and the STD over
                        realizations. The exact reference appears as
                        series="reference" (std empty). Tidy format pivots
                        directly in Excel: e.g. filter observable=energy,
                        pivot series against time.

  <stem>_summary.csv    The scalar tables: per-M costs and errors, per-ntraj
                        mcsolve stats, per-dimension timings -- whatever the
                        file contains.

Run:  python export_csv.py              (exports every data/*.json)
      python export_csv.py cost_scaling_spin_chain.json   (just one)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from common import DATA_DIR

CSV_DIR = DATA_DIR / "csv"

DYN_COLS = ["system", "dim", "observable", "series", "time", "mean", "std"]


def _tlist(doc):
    t = doc["meta"]["tlist"]
    return np.linspace(t["t0"], t["t1"], t["n"])


def _writer(path, cols):
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(cols)
    return fh, w


def _dyn_rows(w, system, dim, observable, series, tlist, mean, std=None):
    mean = np.asarray(mean, dtype=float)
    std = None if std is None else np.asarray(std, dtype=float)
    for i, t in enumerate(tlist):
        w.writerow([system, dim, observable, series, f"{t:.6g}",
                    f"{mean[i]:.10g}", "" if std is None else f"{std[i]:.10g}"])


def _sweep_dynamics(w, system, dim, observable, tlist, sweep, key):
    for row in sweep:
        s = np.asarray(row[key], dtype=float)
        _dyn_rows(w, system, dim, observable, f"SLB M={row['M']}", tlist,
                  s.mean(axis=0), s.std(axis=0, ddof=1))


# ---------------------------------------------------------------------------
# one exporter per data-file family
# ---------------------------------------------------------------------------
def export_accuracy_vs_M(doc, stem):
    system = doc["meta"]["params"]["system"]
    tlist = _tlist(doc)
    fh, w = _writer(CSV_DIR / f"{stem}_dynamics.csv", DYN_COLS)
    for obs, refkey, key in (("energy", "reference_energy", "samples_energy"),
                             ("coherence", "reference_coherence",
                              "samples_coherence")):
        _dyn_rows(w, system, doc["dim"], obs, "reference", tlist, doc[refkey])
        _sweep_dynamics(w, system, doc["dim"], obs, tlist, doc["slb_sweep"], key)
    fh.close()

    fh, w = _writer(CSV_DIR / f"{stem}_summary.csv",
                    ["system", "dim", "n_l", "M", "ensemble_cost_s",
                     "t_davies_s", "t_reference_s"])
    for row in doc["slb_sweep"]:
        w.writerow([system, doc["dim"], doc["n_l"], row["M"],
                    f"{row['cost']:.6g}", f"{doc['t_davies']:.6g}",
                    f"{doc['t_reference']:.6g}"])
    fh.close()
    return 2


def export_frontier(doc, stem):
    system = doc["meta"]["params"]["system"]
    tlist = _tlist(doc)
    fh, w = _writer(CSV_DIR / f"{stem}_dynamics.csv", DYN_COLS)
    _dyn_rows(w, system, doc["dim"], "energy", "reference", tlist,
              doc["reference"])
    _sweep_dynamics(w, system, doc["dim"], "energy", tlist, doc["slb_sweep"],
                    "samples")
    fh.close()

    fh, w = _writer(CSV_DIR / f"{stem}_summary.csv",
                    ["system", "dim", "method", "knob", "value",
                     "cost_s", "bias", "sem", "rmse"])
    for row in doc["slb_sweep"]:
        w.writerow([system, doc["dim"], "SLB", "M", row["M"],
                    f"{row['per_run_cost']:.6g}", "", "", ""])
    for r in doc["mc"]:
        w.writerow([system, doc["dim"], "mcsolve", "ntraj", r["ntraj"],
                    f"{r['cost']:.6g}", f"{r['bias']:.6g}",
                    f"{r['sem']:.6g}", f"{r['rmse']:.6g}"])
    fh.close()
    return 2


def export_isocost_vs_dim(doc, stem):
    system = doc["meta"]["params"]["system"]
    tlist = _tlist(doc)
    fh, w = _writer(CSV_DIR / f"{stem}_dynamics.csv", DYN_COLS)
    for p in doc["points"]:
        _dyn_rows(w, system, p["dim"], "energy", "reference", tlist,
                  p["reference"])
        _sweep_dynamics(w, system, p["dim"], "energy", tlist, p["slb_sweep"],
                        "samples")
    fh.close()

    fh, w = _writer(CSV_DIR / f"{stem}_summary.csv",
                    ["system", "dim", "n_l", "method", "knob", "value",
                     "per_run_or_per_traj_s", "rmse_repeats"])
    for p in doc["points"]:
        for row in p["slb_sweep"]:
            w.writerow([system, p["dim"], p["n_l"], "SLB", "M", row["M"],
                        f"{row['per_run_cost']:.6g}", ""])
        for r in p["mc_fit"]:
            w.writerow([system, p["dim"], p["n_l"], "mcsolve", "ntraj",
                        r["ntraj"], f"{r['per_traj_time']:.6g}",
                        " ".join(f"{x:.6g}" for x in r["rmse_repeats"])])
    fh.close()
    return 2


def export_cost_scaling(doc, stem):
    system = doc["meta"]["params"]["system"]
    tlist = _tlist(doc)
    fh, w = _writer(CSV_DIR / f"{stem}_dynamics.csv", DYN_COLS)
    for p in doc["points"]:
        if p.get("reference") is not None:
            _dyn_rows(w, system, p["dim"], "energy", "reference", tlist,
                      p["reference"])
    fh.close()

    fh, w = _writer(CSV_DIR / f"{stem}_summary.csv",
                    ["system", "dim", "n_l", "t_davies_s", "t_full_s",
                     "t_slb_fixed_s", "M", "rmse", "rmse_std", "sweep_cost_s",
                     "mse", "sem_sq", "diverged"])
    def _g(row, key):
        v = row.get(key)
        return "" if v is None else f"{v:.6g}"
    for p in doc["points"]:
        base = [system, p["dim"], p["n_l"],
                "" if p.get("t_davies") is None else f"{p['t_davies']:.6g}",
                "" if p["t_full"] is None else f"{p['t_full']:.6g}",
                f"{p['t_slb_fixed']:.6g}"]
        if p["m_sweep"]:
            for row in p["m_sweep"]:
                w.writerow(base + [row["M"], _g(row, "rmse"),
                                   _g(row, "rmse_std"), _g(row, "cost"),
                                   _g(row, "mse"), _g(row, "sem_sq"),
                                   "yes" if row.get("diverged") else ""])
        else:
            w.writerow(base + [""] * 7)
    fh.close()
    return 2


EXPORTERS = {
    "accuracy_vs_M_": export_accuracy_vs_M,
    "frontier_": export_frontier,
    "isocost_vs_dim_": export_isocost_vs_dim,
    "cost_scaling_": export_cost_scaling,
}


def main():
    import json
    targets = ([DATA_DIR / a for a in sys.argv[1:]] if len(sys.argv) > 1
               else sorted(DATA_DIR.glob("*.json")))
    if not targets:
        raise SystemExit(f"no data files in {DATA_DIR} - run the run_*.py "
                         f"scripts first")
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for path in targets:
        stem = path.stem
        exporter = next((fn for prefix, fn in EXPORTERS.items()
                         if stem.startswith(prefix)), None)
        if exporter is None:
            print(f"  {path.name}: no exporter for this family, skipped")
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        n = exporter(doc, stem)
        print(f"  {path.name} -> {n} CSVs in {CSV_DIR}")


if __name__ == "__main__":
    main()
