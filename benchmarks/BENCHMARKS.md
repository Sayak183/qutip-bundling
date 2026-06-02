# Benchmarks

This page documents the performance of `qutip-bundling` against the standard
QuTiP solvers. All numbers below were produced by the two scripts in this
folder:

- `benchmark_scaling.py` — cost versus system size, and accuracy versus the
  bundle size *M*.
- `benchmark_vs_mcsolve.py` — accuracy-versus-cost frontier against QuTiP's
  Monte-Carlo trajectory solver `mcsolve`.

Both scripts are self-contained: `pip install qutip-bundling matplotlib`, then
`python benchmark_scaling.py`. Re-running them regenerates every figure here.

## What is being measured

A Lindblad master equation with a large number `N_L` of collapse operators is
expensive: the dissipator costs one matrix product per operator, and `N_L`
typically grows as `N^2` in the Hilbert-space dimension `N`, so propagating the
full equation scales as roughly `O(N^5)` per step. The bundling method replaces
the `N_L` operators with `M` random *bundled* combinations whose dissipator
equals the full one in expectation. With `M` held fixed as the system grows,
the per-step cost drops to `O(N^3)`.

The benchmarks therefore ask two distinct questions, and it is worth keeping
them separate:

1. **Does it scale?** As the system grows, does the wall-clock cost of bundling
   really grow more slowly than the full master equation — and where is the
   crossover below which the full solve is simply the better choice?
2. **Is it accurate, and at what price?** Bundling is a stochastic
   approximation with a tunable knob `M`. More bundles means less error but more
   work. The honest way to judge a stochastic method is not raw speed but the
   error it achieves for a given amount of compute, compared against the other
   stochastic option a user already has — `mcsolve`.

## The two test systems

The benchmarks use two systems chosen to bracket the kinds of problems users
bring to a Lindblad solver. Both relax toward thermal equilibrium under the
same detailed-balance ohmic bath spectral function `gamma(omega)`, and in both
cases the collapse operators are built with `davies_operators(H, X, gamma)`,
which diagonalizes `H`, forms one Bohr-frequency operator per pair of energy
levels, and weights each by the bath response at that frequency.

### Spin chain

A dissipative transverse-field Ising chain of `n` spins:

```
H = -J * sum_i  sigma_z(i) sigma_z(i+1)   -   h * sum_i sigma_x(i)
```

with `J = 1.0`, `h = 0.6`. The bath couples to the total transverse
magnetization `X = sum_i sigma_x(i)`, and the chain starts fully polarized. The
system size is set by the number of spins, so the Hilbert dimension is `2^n`.
Because the energy eigenbasis mixes all the sites, essentially every pair of
levels contributes a Davies operator, and `N_L` climbs steeply with size — 64
operators at dimension 16, over 2000 at dimension 128. This is a recognizable
model across condensed-matter and quantum-computing work, and it is bundling's
natural home: a large operator count that grows quickly with system size.

### Oscillator + bath

An anharmonic oscillator coupled to a two-level spin, with the oscillator
position coupling to the bath:

```
H = omega0 * (n + 1/2)  +  anh * n^2  +  (spin_gap/2) * sigma_z  +  coupling * x * sigma_x
```

with `omega0 = 1.0`, `anh = 0.1`, `spin_gap = 1.0`, `coupling = 0.3`. The bath
couples through the oscillator position `X = x`, and the system size is set by
the Fock-space truncation. This system is close to the molecular/vibronic
physics the method was originally developed for, so it shows how bundling
behaves on a realistic problem rather than only on an idealized chain.

## Result 1 — cost versus system size

![spin chain scaling](benchmark_scaling_spin_chain.png)

![oscillator scaling](benchmark_scaling_oscillator_bath.png)

Each point is annotated with `N_L`, the number of Lindblad operators at that
size. The red curve (full `mesolve`, paying for all `N_L` operators) is faster
for the smallest systems, where the sampling overhead of bundling does not yet
pay off. Its cost then rises steeply and reaches a point — marked by the dashed
line — past which a single solve exceeds the time/memory budget. The green
curve (bundling, using only `M = 4` of the `N_L` operators) starts higher but
rises far more gently and continues well past where the full solve stops.

The practical reading: **below the crossover, use the full master equation;
above it, bundling is what lets the calculation finish at all.** On the spin
chain the full solve at dimension 32 (218 operators) takes around a minute,
versus under a second for bundling — roughly an 80× speedup at that size — and
at dimension 128 (≈2200 operators) the full solve is out of reach while bundling
completes in seconds. The oscillator shows the same pattern even more sharply:
at dimension 32 the full solve takes several minutes where bundling finishes in
about a second.

## Result 2 — accuracy versus the bundle size M

![spin chain accuracy](benchmark_accuracy_spin_chain.png)

![oscillator accuracy](benchmark_accuracy_oscillator_bath.png)

These plot the energy expectation `<H(t)>` against the full Lindblad reference
(black) as the system relaxes. The shaded band is one standard deviation over
stochastic realizations. As `M` grows the bundled mean tightens onto the
reference and the band narrows — the approximation is not a fixed compromise but
a dial the user controls.

The two systems also differ in how cleanly the bundled mean tracks the
reference at small `M`. On the spin chain the bias and spread at `M = 2` are
clearly visible and shrink steadily as `M` grows. On the oscillator the bundled
mean already sits essentially on the reference at `M = 2` — the residual error
there is roughly an order of magnitude smaller at the same `M` (see the frontier
numbers below). How quickly bundling converges in `M` is therefore
system-dependent: it is set by the spread of the individual operator
contributions to the dissipator, not by the Hilbert-space dimension alone, so it
is worth checking the convergence on your own system rather than assuming a
fixed `M` is enough.

## Result 3 — accuracy-versus-cost against mcsolve

![spin chain frontier](benchmark_frontier_spin_chain.png)

![oscillator frontier](benchmark_frontier_oscillator_bath.png)

This is the comparison against the other stochastic option, QuTiP's
quantum-trajectory solver `mcsolve`. Both methods have an accuracy knob —
bundle size `M` for bundling, trajectory count `ntraj` for `mcsolve` — so
neither raw speed nor raw accuracy alone is a fair summary. Each curve sweeps
its own knob; the axes are wall-clock time (cheaper to the left) and
max-over-time error against the full reference (better at the bottom), so the
method sitting toward the **lower-left wins at matched accuracy**. Error bars
are the standard error over independent repeats.

On the spin chain, bundling sits below `mcsolve` across the useful range: to
reach an error near 0.02 it needs well under a second, where `mcsolve` needs
several seconds of trajectories — roughly an order of magnitude cheaper at
matched accuracy. (At the very loosest accuracy a handful of trajectories is
the single cheapest point, so for a quick rough look `mcsolve` is fine.) On the
oscillator the gap is much larger: bundling reaches errors around `10^-3` in
under a second, while `mcsolve` after a thousand trajectories and roughly 20
seconds is still near `10^-1` — about two orders of magnitude less accurate at
far higher cost. This is the regime bundling is built for, and it is where the
method most clearly earns its place.

## Reproducing and reading these numbers

The absolute times depend on the machine, the CPU core count, and the BLAS
build, so treat them as relative comparisons rather than fixed constants. A few
notes for anyone re-running:

- `mcsolve` parallelizes its trajectories across CPU cores, so its position on
  the frontier shifts with the number of cores available; state the core count
  when reporting.
- The first solve of any method pays one-time import and compilation costs; for
  careful timing, discard a warm-up run.
- The system size, time grid, and tolerances are identical across all methods
  in a given figure, so the comparison within each plot is apples-to-apples.
