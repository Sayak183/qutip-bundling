# Benchmarks

This page benchmarks `qutip-bundling` (stochastic Lindblad bundling, **SLB**)
against the two standard QuTiP solvers it competes with: the exact Lindblad
master equation `mesolve`, and the Monte-Carlo trajectory solver `mcsolve`.

Everything below is produced by two self-contained scripts in this folder:

- `benchmark_scaling.py` — cost and accuracy versus system size.
- `benchmark_vs_mcsolve.py` — the accuracy-versus-cost frontier against `mcsolve`.

To regenerate every figure: `pip install qutip-bundling matplotlib`, then run each
script. The supporting checks (`benchmark_convergence.py`,
`benchmark_jackknife.py`, `benchmark_seed_robustness.py`) produce the Result 4
figures.

---

## 1. The core idea, and the two methods being compared

A Lindblad master equation with many collapse operators is expensive. The
dissipator costs one matrix product per operator, and the number of operators
$N_L$ usually grows like $N^2$ in the Hilbert-space dimension $N$, so a full
solve scales as roughly $O(N^5)$ per step. **SLB** replaces the $N_L$ operators
with $M$ random *bundled* combinations whose dissipator equals the full one in
expectation; with $M$ held fixed as the system grows, the per-step cost drops to
$O(N^3)$.

There are two stochastic methods on the table, and the single most important
thing to understand up front is that **they randomize different things.** This
is why their costs and errors behave so differently, and why the comparison has
to be empirical.

**`mcsolve` randomizes the state.** It *unravels* the master equation into
random pure-state trajectories. One trajectory is a wavefunction
$|\psi(t)\rangle$ that drifts under the non-Hermitian effective Hamiltonian
$H_{\rm eff} = H - \tfrac{i}{2}\sum_a L_a^\dagger L_a$, interrupted by random
*quantum jumps*: at random times one of the original $N_L$ collapse operators
$L_a$ fires (chosen with probability $\propto\langle\psi|L_a^\dagger
L_a|\psi\rangle$) and the state resets to $L_a|\psi\rangle$. A single trajectory
looks nothing like the smooth answer; you recover the density matrix by
averaging over `ntraj` trajectories. Both the jump times and which operator fires are random, so no two trajectories are alike — they differ even in how many jumps occur. Raising ntraj does not change any single trajectory; it only adds more independent samples to the average, shrinking the Monte-Carlo error as $N_{\rm traj}^{-1/2}$.  **The randomness is in the state path, and
all $N_L$ operators are kept exact.**

**SLB randomizes the operators.** It keeps the full density matrix and runs an
ordinary deterministic Lindblad evolution — but with the $N_L$ collapse
operators replaced by $M$ random bundles. No jumps, no wavefunctions, just a
density-matrix ODE with fewer operators, averaged over a handful of
realizations. **The randomness is in the operators, and the full state is
kept.**

So the two attack the cost from opposite directions. `mcsolve` propagates cheap
$N$-vectors but touches all $N_L$ operators and needs many trajectories to
suppress its noise. SLB propagates the more expensive $N\times N$ density matrix
but with only $M\ll N_L$ operators and few realizations. Which wins depends on
$N$, $N_L$, $M$, and `ntraj` together — hence the empirical comparison in the
rest of this page.

A consequence worth stating now: the two methods have different **error knobs**,
and they are *not* symmetric (see §4). `mcsolve` has one knob (`ntraj`) that
controls pure statistical noise. SLB has two knobs (`M` and the number of
realizations) controlling two different error sources (a bias and a
fluctuation). Keeping that asymmetry in mind makes every results section below
easier to read.

---

## 2. The two test systems (fully specified)

Both systems are weakly coupled to the **same** thermal bath and relax toward
thermal equilibrium. In both, the collapse operators are built with
`davies_operators(H, X, gamma)`, which diagonalizes the system Hamiltonian $H$,
forms one Bohr-frequency operator $|a\rangle\langle b|$ per pair of energy
levels $(a,b)$ with $\omega_{ab}=E_b-E_a$, and weights each by the bath response
$\sqrt{\gamma(\omega_{ab})}$ at that transition frequency.

### 2.1 The bath (shared by both systems)

The bath is specified entirely by one spectral function — the rate at which the
bath exchanges energy quantum $\omega$ with the system:

$$
\gamma(\omega) = \alpha\\omega*e^{-|\omega|/\omega_c}\\big/\\big(1-e^{-\omega/k_BT}\big),
\qquad \alpha = 0.3,\; k_BT = 0.5,\; \omega_c = 8 .
$$

Reading the three factors:

- $\alpha\,\omega$ — an **ohmic** spectral density (linear in $\omega$ at low
  frequency). $\alpha=0.3$ sets the overall system–bath coupling strength.
- $e^{-|\omega|/\omega_c}$ — an exponential high-frequency cutoff at
  $\omega_c=8$. Since the systems' transition frequencies are of order $1$, the
  cutoff sits well above them and the bath is effectively broadband across the
  transitions that matter.
- $1/(1-e^{-\omega/k_BT})$ — the thermal occupation factor at temperature
  $k_BT=0.5$. It enforces **detailed balance** (the KMS condition
  $\gamma(-\omega)/\gamma(\omega)=e^{-\omega/k_BT}$), which is what guarantees
  relaxation toward the Gibbs state rather than runaway heating. At $\omega\to0$
  this factor gives the finite limit $\gamma(0)=\alpha\,k_BT$.

So in one line: **an ohmic bath with an exponential cutoff, at temperature
$k_BT=0.5$, satisfying detailed balance.**

**Building the Lindblad operators.** Both systems turn $(H_{\rm sys}, X, \gamma)$
into collapse operators by the same Davies (secular) recipe —
`davies_operators(H, X, gamma)`. First diagonalize the system Hamiltonian,

$$H_{\rm sys}\,|a\rangle = E_a\|a\rangle .$$

Then every **ordered** pair of eigenstates $(a,b)$ whose coupling element
$\langle a|X|b\rangle$ is non-zero contributes one Lindblad operator, tagged with
the Bohr frequency of that transition:

$$\omega_{ab} = E_b - E_a , \qquad
  c_{ab} = \sqrt{\gamma(\omega_{ab})}\\langle a|X|b\rangle\|a\rangle\langle b| .$$

The number of such operators is $N_L$ — one per energy-conserving channel the
coupling opens. The sign convention $\omega_{ab}=E_b-E_a$ is what makes the
dynamics relax rather than heat up: a downward transition ($E_b>E_a$, so $b$ is
the higher level) carries $\omega_{ab}>0$, where detailed balance makes
$\gamma$ largest, so energy is preferentially emitted to the bath and the state
flows toward $\rho_\infty\propto e^{-H_{\rm sys}/k_BT}$. (`davies_operators`
bakes in this convention; building the operators by hand with the opposite sign
runs the system uphill.)

The two systems feed *different* $(H_{\rm sys}, X)$ into this one recipe:

- **System A** (§2.3): $X = \sum_i \sigma^x_i$, the collective transverse
  magnetization. Diagonalizing the 16-state chain and keeping every pair with
  $\langle a|X|b\rangle\neq 0$ gives $N_L \approx 64$. (The exact count is mildly
  sensitive to how the chain's symmetry degeneracies are resolved numerically —
  62–64 depending on the linear-algebra backend — because within a degenerate
  energy level the eigenbasis is not unique.)
- **System B** (§2.4): $X = x\otimes I$, the oscillator position. The anharmonic
  ladder is non-degenerate, so the count is exact and basis-independent:
  $N_L = 128$.

In code this is a single call per system:

```python
H, X, psi0 = build_spin_chain(4)        # System A  (build_oscillator_bath(8) for B)
c_ops = davies_operators(H, X, gamma)   # the {c_ab} above, length N_L
```

### 2.2 Is this weak coupling? Yes — in both senses.

1. **By construction.** `davies_operators` builds a Davies/secular master
   equation, which is *derived* in the weak system–bath coupling (Born–Markov)
   limit. Using the Davies operators places the model in the weak-coupling
   Lindblad regime by assumption — that is the theory's domain of validity.
2. **By the numbers.** The bath coupling scale $\alpha=0.3$ is smaller than each
   system's coherent energy scales ($J=1$ for the chain, $\omega_0=1$ for the
   oscillator), so dissipation is slower than the internal coherent dynamics —
   the weak-coupling ordering. It is "moderate" weak coupling: strong enough to
   produce real relaxation over $t\in[0,5]$, not so strong that the perturbative
   description breaks.

### 2.3 System A — dissipative transverse-field Ising chain

The **system Hamiltonian** for $n$ spins is

$$
H_{\rm sys} = -J\sum_{i=1}^{n-1}\sigma^z_i\sigma^z_{i+1}\-h\sum_{i=1}^{n}\sigma^x_i,
\qquad J = 1.0,\; h = 0.6 .
$$

The first term is nearest-neighbour Ising coupling; the second is a transverse
field. The bath couples to the system through the **coupling operator**
$X = \sum_i \sigma^x_i$ (total transverse magnetization): the reservoir acts on
the chain *via* this observable, driving transitions that relax its magnetization
toward thermal equilibrium. This is a single *collective* coupling — all $n$ spins
share **one** common bath through this global operator, rather than each spin
relaxing into its own independent reservoir. The chain starts fully polarized,
$|\psi_0\rangle = |{\uparrow\uparrow\cdots\uparrow}\rangle$. Starting from this pure,
fully ordered state, the open evolution relaxes the chain to the thermal Gibbs
state $\rho_\infty \propto e^{-H_{\rm sys}/k_BT}$ at the bath temperature
$k_BT = 0.5$: the net magnetization decays, the initial coherence is lost, and
energy flows out into the bath.

![System A schematic](system_a_schematic.png)

System A: a transverse-field Ising chain ($J = 1.0$, $h = 0.6$), fully
polarized at $t = 0$, coupled through the global operator
$X = \sum_i \sigma^x_i$ to a single collective ohmic bath.

Because there is no separate bath Hilbert space in the Lindblad description, the
**total object being evolved** is the master equation
$\dot\rho = -i[H_{\rm sys},\rho] + \sum_a \mathcal{D}[c_a]\rho$, with the
dissipators $c_a$ generated from $(H_{\rm sys}, X, \gamma)$ as in §2.1. The
Hilbert dimension is $2^n$. The energy eigenbasis mixes all sites, so nearly
every level pair contributes a Davies operator and $N_L$ climbs steeply with
size — 4 operators at $n=2$, $\sim 64$ at $n=4$ (dim 16), $\sim 213$ at $n=5$ (dim 32),
$\sim 2200$ at $n=7$ (dim 128). This rapid operator growth is SLB's natural home.

### 2.4 System B — anharmonic oscillator coupled to a spin

The **system Hamiltonian** is

$$
H_{\rm sys} = \omega_0\\left(n+\tfrac12\right) + \chi*n^2
            + \tfrac{\Delta}{2}\\sigma_z + g\(x\otimes\sigma_x)
$$

with $\omega_0=1.0$, anharmonicity $\chi=0.1$, spin gap $\Delta=1.0$, and an
internal oscillator–spin coupling $g=0.3$. Here $n=a^\dagger a$ is the number
operator and $x=(a+a^\dagger)/\sqrt2$ the position. The four terms are: the bare
oscillator, its anharmonicity, the spin's energy splitting, and a coherent
oscillator–spin coupling.

![System B schematic](system_b_schematic.png)

System B: an anharmonic oscillator whose energy gaps widen up the ladder,
coupled to a two-level spin by an internal coherent coupling $g$. A single
ohmic bath couples to the oscillator position $X = x\otimes I$ only, so the
spin relaxes solely indirectly through $g$. The oscillator starts in its top
Fock level.

The bath couples to the system through the **coupling operator** $X = x\otimes I$
(the oscillator position): a phonon/photon-like reservoir acts *via* this observable,
damping the oscillator's motion and draining its energy toward thermal equilibrium.
It is a single shared bath — $X = x\otimes I$ acts only on the oscillator, so the
oscillator and spin couple to **one** common reservoir (the spin has no separate
bath of its own). Because $X$ touches only the oscillator, the bath never damps the
spin directly; dissipation reaches the spin only indirectly, through the internal
coherent coupling $g\(x\otimes\sigma_x)$. The system starts in the oscillator's
top Fock state with the spin down. As above, the total evolved object is the Lindblad master
equation with dissipators built from $(H_{\rm sys}, X, \gamma)$ as in §2.1 ($N_L = 128$ at dim 16). Note the two
distinct "couplings": $g=0.3$ is an **internal coherent** coupling inside
$H_{\rm sys}$, whereas $\alpha=0.3$ is the **system–bath** coupling carried by
$\gamma$ through $X$ — they are different physics that happen to share a value.
The size is set by the Fock truncation. This system is close to the
molecular/vibronic problems the method was developed for.

---

## 3. What we measure, and how the error is reported

### 3.1 Error: a time-resolved band, and the single numbers from it

The quantity of interest is how well the bundled $\langle H(t)\rangle$ (and, in
Result 4, a coherence) tracks the exact reference. The accuracy and coherence
figures show this **resolved over the whole trajectory**: for each $M$, the SLB
mean curve is drawn with a shaded **$\pm1$ standard-deviation band** (the spread
over the stochastic realizations) and, beneath it, a **residual panel**
$\langle H\rangle_{\rm SLB}-\langle H\rangle_{\rm ref}$. Nothing is collapsed to a
single instant — the error is visible at every time.

This keeps the **two error components** separate:

- the **bias** is how far the residual curve sits from zero — the systematic
  offset of the bundled mean from the exact answer;
- the **statistical fluctuation** is the width of the shaded band — how much a
  single bundled run scatters around that mean.

The distinction matters because one combined number hides which dominates, and
at small $M$ or small systems the fluctuation can be *larger* than the bias.

When the error must instead live on an axis — the **error-vs-X figures** — it is
collapsed to one scalar per point, and the right scalar depends on the figure's
job:

- **Characterizing one method's own scaling** — convergence vs $M$ and the
  jackknife vs dimension (both Result 4) — uses the **max-over-time** error, the
  worst deviation over the trajectory. For "how fast does this method's error
  shrink," the conservative worst case is the robust diagnostic and needs no
  chosen time.
- **Comparing two methods head-to-head** — the SLB-vs-`mcsolve` frontier
  (Result 3) and the substep integrator check (Result 4) — uses the error at a
  single **mid-relaxation time $t=2.5$**. Here a fixed representative time is
  *fairer* than max-over-time: the maximum of noisy samples is biased upward, so
  a worst-case metric would inflate the noisier method's apparent error.

(The dynamics run to $t=5$ in natural units — $J=1$ for the chain, $\omega_0=1$
for the oscillator — over 40 output points; $t=2.5$ is the mid-relaxation
sample, where the energy has substantially decayed but not yet saturated.)

### 3.2 How much sampling each method does (read this — it is the confusing part)

The two methods' "sample counts" are not the same kind of thing, so here they
are explicitly.

**SLB has two levels of sampling:**

- *Within one solve:* $M$ random bundled operators. Each solve is already a
  Monte-Carlo average over $M$ random draws (that is the unbiased-estimator
  property).
- *Across solves:* the reported mean and its std come from averaging
  `n_realizations` independent bundled solves, each with a fresh random draw.
  This is the closer analogue of `ntraj`.

So a single reported SLB point is built from `n_realizations` full
density-matrix solves of $M$ operators each — a total of $M\times$
`n_realizations` random draws. The values used here:

| figure | `M` | `n_realizations` | outer repeats (for error bars) |
|---|---|---|---|
| scaling (Result 1) | 2, 4, 8 | 8 | 1 |
| accuracy / coherence (Result 2, 4) | system-dependent | 32 | 1 |
| frontier (Result 3) | 1, 2, 4, 8, 16 | 16 | a few |

**`mcsolve` has one level of sampling:** a single reported point is `ntraj`
independent trajectories (swept over `[50, 200, 1000]` in scaling, and across a
similar range in the frontier), run single-threaded so its wall-clock time is
the full sequential cost of all trajectories — matching SLB's single-threaded
realization loop.

### 3.3 Integrators: matched where it is possible, disclosed where it is not

The full `mesolve` reference and `mcsolve` both use QuTiP's **adaptive**
integrator at stated tolerances (`atol=1e-8`, `rtol=1e-6`): they choose their
own step sizes to hit an error target, so there is no single step size to quote.
SLB's native backend uses **fixed-step RK4** with a small number of substeps per
output step (4 here; the result is already converged by 2).

This is a deliberate, disclosed asymmetry, not a hidden advantage:

- All methods share the **same output time grid** and the **same exact
  reference**, so they are compared at identical points.
- You *can* match the full `mesolve` reference to SLB by running the full
  (unbundled) operators through the same fixed-step RK4. Doing so isolates the
  integration error, which falls far below the bundling error — orders of
  magnitude below for these systems (the substep-convergence check in
  **Result 4** makes this explicit) — so it does not contaminate the comparison.
- You *cannot* match `mcsolve` this way: its accuracy knob is `ntraj`, not a step
  size, so a shared accurate reference is the only sensible common ground. SLB is
  therefore not winning by integrating more loosely — if anything `mcsolve`
  integrates each trajectory more tightly; SLB wins by needing far fewer
  operators and samples.

---

## 4. How `mcsolve`'s error works, versus SLB's

This is the asymmetry promised in §1, made precise.

**`mcsolve`: essentially no bias; error is pure fluctuation.** Because the
observable is linear in $\rho$ and the unraveling is exact in expectation
($\mathbb{E}[|\psi\rangle\langle\psi|]=\rho$), the trajectory average is an
**unbiased** estimator of the true expectation for *any* `ntraj`. Increasing
`ntraj` does not move the mean — it only shrinks the statistical scatter, as

$$
\text{StdErr} = \frac{\sigma_{\rm traj}}{\sqrt{\text{ntraj}}} .
$$

The fluctuation depends on **`ntraj`** (the explicit knob, $1/\sqrt{\text{ntraj}}$)
and on **$\sigma_{\rm traj}$**, the intrinsic per-trajectory spread — large for
strongly dissipative systems, many jump channels, long times, and observables
sensitive to *which* jumps occurred (coherences especially). The only
non-statistical error floor is ODE integration error, set by the tolerances —
not by `ntraj`. **To reduce `mcsolve` error: raise `ntraj`** (and the floor is
governed by the tolerances).

**SLB: a bias *and* a fluctuation — two knobs.** The randomness sits in the
dissipator, and $\rho$ is a *nonlinear* function of the generator, so even
though the bundled dissipator is unbiased, pushing its noise through the
nonlinear evolution leaves a **finite-$M$ bias** of order $1/M$ in the state.
On top of that sits a run-to-run **fluctuation** that averages down over
realizations. So:

- **Bias** $\sim M^{-1}$ — reduced by raising `M` (or removed to leading order by
  the built-in jackknife correction). This is what grows with system size at
  fixed `M` (more operators compressed into the same bundles — see Result 1).
- **Fluctuation** — reduced by raising `n_realizations` (and also falls with
  `M`).

The practical upshot: `mcsolve` users turn one knob (`ntraj`) against noise; SLB
users turn `M` against bias and `n_realizations` against fluctuation. The
frontier (Result 3) sweeps the bias knob `M` for SLB against the noise knob
`ntraj` for `mcsolve`.

---

## 5. Results

### Result 1 — cost scaling versus the exact solver

![spin chain cost scaling](benchmark_cost_scaling_spin_chain.png)
![oscillator cost scaling](benchmark_cost_scaling_oscillator_bath.png)

Wall-clock time for one solve versus Hilbert-space dimension $N$, comparing the
exact full-dissipator solver against SLB. Each point is a single timed solve:
one `mesolve` with all $N_L$ collapse operators, and one SLB realization at a
representative bundle size ($M=8$). The dashed line marks where one full
`mesolve` exceeds the time budget — past it the exact solver is impractical.

**The two scalings.** Full `mesolve` evolves the density matrix with the
complete dissipator, an operation count that grows like $O(N^5)$; the measured
large-$N$ slope is consistent with that (the legend reports the fitted
exponent). SLB only ever propagates $M$ bundled operators, so its cost grows
like the underlying dense propagation, $O(N^3)$ in operation count; the measured
large-$N$ slope is shallower still ($\sim N^2$ over the range shown, where dense
linear algebra is not yet the sole bottleneck). The practical consequence is the
gap that opens up with size: `mesolve` is cheapest only on the smallest systems,
then climbs steeply and stops at the wall, while SLB continues cheaply to
dimensions the exact solver cannot reach. (Small dimensions are dominated by
fixed overhead, not the asymptotic operation count, so the slope is fit from the
large-$N$ end.)

**What this figure does and doesn't show.** It shows SLB's speedup over the
*exact* solver — the comparison the method is built to win. It leaves `mcsolve`
out on purpose: a trajectory method's cost scales differently, so a
cost-versus-size plot is the wrong way to judge it. The fair SLB-versus-`mcsolve`
question is *accuracy per unit cost* — to match SLB's accuracy, `mcsolve` needs
many trajectories — and that comparison is **Result 3**.

**A practical note.** On the stiffer oscillator at large sizes, SLB's RK4
integrator needs a few substeps per step to stay stable — one substep diverges.
These runs use a small fixed substep count (given in the caption) inside the
stable range. If the integration ever blows up to a non-finite state, the solver
raises `SolverInstabilityError` instead of silently returning a corrupted result.

### Result 2 — accuracy versus the bundle size $M$

![spin chain accuracy](benchmark_accuracy_spin_chain.png)
![oscillator accuracy](benchmark_accuracy_oscillator_bath.png)

These plot $\langle H(t)\rangle$ against the exact reference (black) as the
system relaxes, with a $\pm1$-std band over realizations. As `M` grows the
bundled mean tightens onto the reference and the band narrows — the
approximation is a dial, not a fixed compromise. The two systems differ in how
fast they converge in `M`: the oscillator already sits essentially on the
reference at `M=2`, while the chain shows a visible bias and spread at `M=2`
that shrink as `M` grows. Convergence speed is set by the spread of the
individual operator contributions, not by dimension alone, so it is worth
checking on your own system.

### Result 3 — accuracy-versus-cost frontier against `mcsolve`

![spin chain frontier](benchmark_frontier_spin_chain.png)
![oscillator frontier](benchmark_frontier_oscillator_bath.png)

Each curve sweeps its own knob (`M` for SLB, `ntraj` for `mcsolve`); the axes
are wall-clock time and error in $\langle H(t)\rangle$ (both lower-is-better), so
the method toward the **lower-left wins at matched accuracy**. Error bars are the
spread over independent repeats. Both methods run at disclosed integration
resolution (§3.3) and share the same grid and reference.

On the spin chain SLB sits below `mcsolve` across most of the range (a few times
cheaper at matched accuracy; at the very loosest accuracy a handful of
trajectories is the single cheapest point). On the oscillator the gap is large:
SLB reaches errors around $10^{-3}$ in a few seconds, while `mcsolve` after a
thousand trajectories is still near $10^{-1}$ — about two orders of magnitude
less accurate at higher cost. This is the regime SLB is built for.

### Result 4 — validation and robustness

The checks that answer the obvious doubts.

**Beyond energy: a coherence.** Energy is nearly diagonal in the energy
eigenbasis, so matching $\langle H\rangle$ says little about off-diagonal
structure. SLB also tracks the most-populated energy-eigenstate coherence
$|a\rangle\langle b|+\text{h.c.}$ with the same convergence in `M`.

![spin chain coherence](benchmark_coherence_spin_chain.png)
![oscillator coherence](benchmark_coherence_oscillator_bath.png)

**Convergence at the predicted rates.** The bias should fall as $M^{-1}$ and the
statistical spread as $M^{-1/2}$.

![spin chain convergence](benchmark_convergence_spin_chain.png)
![oscillator convergence](benchmark_convergence_oscillator_bath.png)

Fitting recovers the predicted $M^{-1}$ bias on both systems ($M^{-0.99}$ chain,
$M^{-0.97}$ oscillator); the spread follows $M^{-1/2}$ on the chain ($M^{-0.52}$)
and faster on the oscillator. Matching the predicted *bias* exponent — the thing
that sets SLB's accuracy — is the strongest single check that the estimator
behaves as derived.

**Bias versus size, with jackknife (this is the Result 1 size trend,
quantified).** At fixed `M` the finite-$M$ bias grows with dimension; the
built-in jackknife correction suppresses it.

![spin chain jackknife](benchmark_jackknife_spin_chain.png)
![oscillator jackknife](benchmark_jackknife_oscillator_bath.png)

The uncorrected bias rises steeply with dimension while the corrected bias stays
comparatively flat — so the Result 1 growth is a known, correctable effect, not
a breakdown. On the oscillator the corrected residual sits at the noise floor
(consistent with zero).

**Seed robustness.** Recomputing the frontier across independent master seeds
leaves the conclusion unchanged: per-seed frontiers cluster tightly and SLB
stays below `mcsolve` for every seed.

![spin chain seed robustness](benchmark_seed_robustness_spin_chain.png)

**Integration robustness — SLB is bundling-limited, not integrator-limited.**
This answers the doubt that SLB might only look fast because it integrates the
master equation more crudely than the adaptive reference. Holding the bundling
fixed (one seed, so the bundles are identical) and sweeping only the RK4 substep
count separates the two error sources: the *pure* integration error — the full,
unbundled operators run through the same RK4 — falls quickly and bottoms out at
the reference tolerance ($\sim 10^{-10}$), while the *total* SLB error stays flat,
set by the bundling ($M$). At 4 substeps the integration error is orders of
magnitude below the bundling error ($\sim 10^{8}\times$ smaller for these systems),
so the substep choice cannot be where SLB's accuracy — or its speed — comes
from; one could integrate far more crudely without moving the SLB error.

The plotted error is the absolute deviation from the adaptive reference at the
fixed mid-point $t=2.5$, $|\langle H\rangle(2.5) - \langle H\rangle_{\rm ref}(2.5)|$
— the same single-time-point metric as the scaling and frontier figures (§3.1).

![spin chain substep convergence](benchmark_substep_convergence_spin_chain.png)
![oscillator substep convergence](benchmark_substep_convergence_oscillator_bath.png)

**`mcsolve` fairness.** In Result 3 `mcsolve` runs single-threaded (matching
SLB's single-threaded loop) at stated tolerances; removing its multi-core
advantage does not change the conclusion.

---

## 6. Reproducing and reading these numbers

Absolute times depend on the machine, core count, and BLAS build — treat them as
relative comparisons. A few notes:

- `mcsolve` parallelizes trajectories across cores; Result 3 pins it
  single-threaded to match SLB. State the core count when reporting.
- The first solve of any method pays one-time import/compile costs; discard a
  warm-up run for careful timing.
- Within each figure the system size, time grid, tolerances, and reference are
  identical across methods, so each plot is internally apples-to-apples.
- The full `mesolve` reference is "exact" only for the Davies–Lindblad model
  defined here; the Davies/secular construction is itself a weak-coupling
  approximation to the underlying open-system dynamics.
