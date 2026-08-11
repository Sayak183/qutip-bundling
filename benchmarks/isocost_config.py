"""Shared configuration for Result 4 (iso-accuracy cost versus dimension).

Edit ``SYSTEM_N_RUNS`` here to change the averaging levels used by both the
data-generation and plotting scripts.  The runner generates enough raw samples
for the largest configured level; the plotter derives every configured level
from those samples.
"""

SYSTEM_N_RUNS = {
    "spin_chain": [4],
    # System B carries the same averaging level as the oscillator so the two
    # large-N_L systems are compared at equal statistical footing; the TFIM
    # chain keeps 4 because its N_L caps M at 31 and the sweep ends early.
    "mixed_chain": [16],
    "oscillator_bath": [16],
}


def run_counts(system):
    """Return validated, strictly increasing run counts for one system."""
    try:
        counts = list(SYSTEM_N_RUNS[system])
    except KeyError as exc:
        raise ValueError(f"no Result 4 run-count configuration for {system!r}") from exc
    if (
        not counts
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in counts)
        or counts != sorted(set(counts))
    ):
        raise ValueError(
            f"SYSTEM_N_RUNS[{system!r}] must contain strictly increasing "
            f"integers >= 2; got {counts!r}"
        )
    return counts
