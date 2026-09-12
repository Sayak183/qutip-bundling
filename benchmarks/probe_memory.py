"""
probe_memory.py
===============

Answer "will the operator list fit?" by measuring, not by extrapolating.

Builds a system's Davies collapse operators exactly as the runners do and
reports the process's peak resident memory beside the operator list's nominal
size. The ratio between them is the number that decides whether a Result 1
job at that size can exist -- and it is not 1.

Why this exists. System B at dimension 256 has a 34 GB operator list, and the
Result 1 job there peaked at 100 GB: 2.9x, from the Davies construction's
intermediates and per-operator Qobj wrapping. Scaled to dimension 512, where
the list is 549 GB, 2.9x is ~1.6 TB against a 1.55 TB node. Whether the real
ratio at 512 is 2.9x or 2.2x is the difference between a three-week job
finishing and being OOM-killed after a week. That is worth two minutes on a
node to find out first.

Run one size or several, ascending. Peak RSS is a process high-water mark, so
after each size the figure reported is that size's own peak provided sizes
ascend. Operators from the previous size are dropped before the next build.

    python probe_memory.py --system mixed_chain --sizes 8 9

On Linux the peak comes from getrusage(RUSAGE_SELF).ru_maxrss. Elsewhere it
falls back to psutil if installed, and otherwise reports only the nominal
size, which is the number this tool exists to distrust.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

from common import (
    build_davies_operators, build_spin_chain, build_mixed_field_chain,
    build_oscillator_bath,
)

BUILDERS = {
    "spin_chain": build_spin_chain,
    "mixed_chain": build_mixed_field_chain,
    "oscillator_bath": build_oscillator_bath,
}


def peak_rss_bytes():
    """Process high-water mark in bytes, or None if it cannot be read."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes; macOS reports bytes.
        return kb * 1024 if sys.platform != "darwin" else kb
    except ImportError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset  # Windows
    except (ImportError, AttributeError):
        return None


def fmt(nbytes):
    if nbytes is None:
        return "n/a"
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if nbytes >= div:
            return f"{nbytes / div:.1f} {unit}"
    return f"{nbytes / 1e3:.0f} KB"


def probe(system, size):
    build = BUILDERS[system]
    t0 = time.perf_counter()
    H, X, _ = build(size)
    dim = H.shape[0]
    baseline = peak_rss_bytes()

    c_ops = build_davies_operators(H, X)
    n_l = len(c_ops)
    elapsed = time.perf_counter() - t0
    nominal = n_l * dim * dim * 16
    peak = peak_rss_bytes()

    print(f"{system}  size {size}  dim {dim}  N_L {n_l:,}   built in {elapsed:.0f} s")
    print(f"   operator list, nominal (N_L x dim^2 x 16 B) : {fmt(nominal):>10}")
    if peak is not None:
        # Subtract what the interpreter and imports already held, so a small
        # size does not report Python's own footprint as a 300x ratio. At
        # the sizes this tool is for, the baseline is noise either way.
        build_cost = max(peak - (baseline or 0), 0)
        print(f"   process peak RSS after the build           : {fmt(peak):>10}")
        print(f"   of which the build itself (peak - baseline): {fmt(build_cost):>10}")
        print(f"   build / nominal                            : {build_cost / nominal:>9.2f}x")
        print(f"   node has 1.55 TB; this leaves              : {fmt(1.55e12 - peak):>10}")
    else:
        print("   peak RSS unavailable on this platform -- run it on the cluster")
    print()

    del c_ops, H, X
    gc.collect()
    return nominal, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    ap.add_argument("--system", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--sizes", type=int, nargs="+", required=True,
                    help="model sizes, ascending (spins for the chains, Fock "
                         "cutoff for the oscillator)")
    args = ap.parse_args()
    if args.sizes != sorted(args.sizes):
        ap.error("--sizes must ascend: peak RSS is a high-water mark, so a "
                 "smaller size after a larger one would report the larger")
    for size in args.sizes:
        probe(args.system, size)


if __name__ == "__main__":
    main()
