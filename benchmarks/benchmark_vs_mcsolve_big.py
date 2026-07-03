"""
benchmark_vs_mcsolve_big.py
===========================

Bigger-system version of the accuracy-vs-cost frontier (Result 3), for a
WORKSTATION -- not the sandbox/CI. It reuses benchmark_vs_mcsolve.py unchanged
and only overrides the configuration, so there is a single source of truth for
the method.

Systems
-------
    * spin_chain, 6 sites  -> dim 64, N_L ~ 866 Lindblad operators
    * oscillator_bath, n_fock=16 -> dim 32

Why this is a separate, heavy script
------------------------------------
SLB itself stays cheap at these sizes -- that is the whole point. The expensive
parts are the *baselines* the benchmark needs:

    * the full mesolve REFERENCE (all N_L operators on a dim-64 Liouville space)
      is what the RMSE is measured against. It needs a lot of RAM and can take
      minutes to tens of minutes.
    * the mcsolve baseline propagates all N_L operators per trajectory, so it is
      slow on the stiff / large-N_L systems (again, minutes).

Guidance:
    * Run on a machine with >= 16 GB RAM; consider running overnight.
    * If the full mesolve reference raises MemoryError, that failure IS the
      motivation for bundling -- but without a reference you cannot measure RMSE
      at that size, so drop to a smaller system for the accuracy frontier and
      use benchmark_cost_scaling.py (which needs no reference beyond the wall)
      to make the large-N cost argument.
    * The substeps guard (inherited from the frontier script) runs first and
      prints OK / WARNING. On larger, stiffer systems the RK4 timestep may need
      raising -- if it WARNS, increase SUBSTEPS in benchmark_vs_mcsolve.py and
      re-run.

Figures are written with distinct names (benchmark_frontier_spin_chain_6spin.png,
benchmark_frontier_oscillator_bath_dim32.png) so the dim-16 figures are kept.

Run:  python benchmark_vs_mcsolve_big.py
"""

from __future__ import annotations

import benchmark_vs_mcsolve as b

# --- bigger systems; distinct names so the small-system figures are preserved ---
b.SYSTEMS = [
    ("spin_chain_6spin",      b.build_spin_chain,      6),   # dim 64, N_L ~ 866
    ("oscillator_bath_dim32", b.build_oscillator_bath, 16),  # dim 32
]

# --- trimmed sweeps so the heavy reference + mcsolve baseline stay tractable ---
# push M higher (N_L is large here) but use a single run count and cap ntraj.
b.M_VALUES = [2, 4, 8, 16, 32, 64]
b.N_RUNS_SWEEP = [16]
b.N_RUNS_MAX = 16
b.NTRAJ_VALUES = [10, 50, 200]     # mcsolve is expensive at dim 64; a few points
                                    # already show it cannot keep up


def _banner():
    print("=" * 72)
    print("BIG-SYSTEM FRONTIER  (workstation job -- not the sandbox / CI)")
    print("  spin_chain 6 sites  -> dim 64, N_L ~ 866")
    print("  oscillator n_fock=16 -> dim 32")
    print("  The full-mesolve reference and mcsolve baseline are the heavy parts")
    print("  (RAM + minutes to tens of minutes each). SLB itself stays cheap.")
    print("  If the reference OOMs, use benchmark_cost_scaling.py for the large-N")
    print("  cost argument instead (it needs no reference past the wall).")
    print("=" * 72)


if __name__ == "__main__":
    _banner()
    b.main()
