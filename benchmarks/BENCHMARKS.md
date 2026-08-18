# Performance Benchmarks: Stochastically Bundled Dissipators

This page benchmarks `qutip-bundling` (stochastic Lindblad bundling, **SLB**)
against the two standard QuTiP solvers it competes with: the exact Lindblad
master equation `mesolve`, and the Monte-Carlo trajectory solver `mcsolve`.

Everything below is produced by self-contained scripts in this folder:

> **New here?** Start with [`README.md`](README.md) to replot a committed result
> in seconds without launching the expensive numerical benchmarks.

- `run_accuracy_vs_M.py` + `plot_accuracy_vs_M.py` — accuracy versus bundle size
  (Result 1). The run script saves the raw per-realization dynamics of both
  observables for every `M` into
  `data/accuracy_vs_M_<system>_dim<D>.json`, timing the Davies-operator
  construction separately from the propagation; the plot script derives the
  mean curves, the bands, and the peak-error decomposition from it.
- `run_cost_scaling.py` + `plot_cost_scaling.py` — cost scaling versus the exact
  solver (Result 2). The exact reference is QuTiP's `mesolve` up to the memory
  wall (~dim 32), and beyond it — where `mesolve` can no longer build its
  superoperator — the native full-dissipator RK4 solver (`rk4_mesolve`), which
  propagates all `N_L` operators directly and is cross-validated against
  `mesolve` wherever both run. The run script does the compute and writes
  `data/cost_scaling_<system>.json` (stamped with package versions, seeds, the
  bundle-size sweep, and the substeps used for each reference); the plot script
  derives the figure from that file in seconds.
- `run_frontier.py` + `plot_frontier.py` — accuracy-versus-cost frontier against
  `mcsolve` (Result 3), same split: raw SLB run samples and per-`ntraj` stats go
  into `data/frontier_<system>_dim<D>.json` (each run also records whether its
  fixed-step integrator stayed stable at the chosen substep count, so an
  under-resolved reference is flagged rather than trusted), and the plot script
  draws the frontier from it. The valid command for the heavy spin-chain
  dimension-64 run is
  `python run_frontier.py --system spin_chain --dims 64 --overwrite`
  ($N_L=113$ in the committed pre-0.6.4 data, 31 once regenerated -- see
  the provenance note in §2.3); preview it first with `--dry-run`.
- `run_isocost_vs_dim.py` + `plot_isocost_vs_dim.py` — iso-accuracy cost versus
  dimension (Result 4), split the same way: the run script writes
  `data/isocost_vs_dim_<system>.json` with the raw run samples and the mcsolve
  $S^2$ fit; the plot script derives $M^\ast$, $\texttt{ntraj}^\ast$, and the
  speedups from it.

To regenerate every figure, install the package (`pip install -e ".[examples,test]"`
from a checkout, or `pip install qutip-bundling matplotlib`) and run each script
from this folder. Start with the `plot_*.py` scripts: they redraw from the saved
`data/*.json` in seconds without recomputing. The `run_*.py` scripts do the
expensive simulations: they require an explicit `--system` or `--all`, support
`--dry-run`, and refuse to replace tracked JSON unless `--overwrite` is given.
The supporting checks (`benchmark_convergence.py`,
`benchmark_jackknife.py`, `benchmark_seed_robustness.py`,
`benchmark_substep_convergence.py`) produce the validation figures (§6).

---

## Contents

**Setup**
- [1. The core idea, and when it applies](#1-the-core-idea-and-when-it-applies)
  - [Will bundling help *your* system?](#will-bundling-help-your-system)
- [2. The three test systems (fully specified)](#2-the-three-test-systems-fully-specified)
  - [2.0 The physical picture, before any equations](#20-the-physical-picture-before-any-equations)
  - [2.1 The bath (shared by all three systems)](#21-the-bath-shared-by-all-three-systems)
  - [2.2 Is this weak coupling? Yes — in both senses.](#22-is-this-weak-coupling-yes--in-both-senses)
  - [2.3 Systems A & B — transverse-field and mixed-field Ising chains](#23-system-a--b--transverse-field-g0-and-mixed-field-g04-ising-chains)
  - [2.4 System C — anharmonic oscillator coupled to a spin](#24-system-c--anharmonic-oscillator-coupled-to-a-spin)
  - [2.5 What each system is for](#25-what-each-system-is-for)
  - [2.6 Worked example: where these numbers actually come from](#26-worked-example-where-these-numbers-actually-come-from)
- [3. What we measure, and how the error is reported](#3-what-we-measure-and-how-the-error-is-reported)
  - [3.1 Which observables, and why not just the energy](#31-which-observables-and-why-not-just-the-energy)
  - [3.2 Error: a time-resolved band, and the single numbers from it](#32-error-a-time-resolved-band-and-the-single-numbers-from-it)
  - [3.3 How much sampling each method does](#33-how-much-sampling-each-method-does)
  - [3.4 Integrators: matched where it is possible, disclosed where it is not](#34-integrators-matched-where-it-is-possible-disclosed-where-it-is-not)
- [4. How `mcsolve`'s error works, versus SLB's](#4-how-mcsolves-error-works-versus-slbs)

**Results**
- [5. Results](#5-results)
  - [5.1 Reading the cost–accuracy plots](#51-reading-the-costaccuracy-plots)
  - [5.2 Memory and stiffness walls](#52-memory-and-stiffness-walls)
  - [5.3 Provenance of Results 1-4](#53-provenance-of-results-1-4)
  - [Result 1 — accuracy versus the bundle size M](#result-1--accuracy-versus-the-bundle-size-m)
  - [Result 2 — cost scaling versus the exact solver](#result-2--cost-scaling-versus-the-exact-solver)
  - [Result 3 — four-method accuracy versus cost](#result-3--four-method-accuracy-versus-cost)
  - [Result 4 — iso-accuracy cost versus dimension](#result-4--iso-accuracy-cost-versus-dimension)
  - [Result 5 — past the reference wall](#result-5--past-the-reference-wall)

**Reference**
- [6. Validation and robustness](#6-validation-and-robustness)
- [7. Reproducing and reading these numbers](#7-reproducing-and-reading-these-numbers)

---

## 1. The core idea, and when it applies

A Lindblad master equation with many collapse operators is expensive: the
dissipator costs one matrix product per operator, so a full solve scales as
roughly $O(N_L N^3)$ per step in the Hilbert-space dimension $N$. **SLB**
replaces the $N_L$ operators with $M$ random *bundled* combinations whose
dissipator equals the full one in expectation, dropping the per-step cost to
$O(M N^3)$.

That statement contains the two conditions that decide whether bundling is effective:

1. **Cost.** Bundling can only pay when $M \ll N_L$. The operator count
   $N_L$ — **not** the Hilbert-space dimension — governs the speedup.
2. **Accuracy.** At a given $M$, accuracy depends on how *local* the collapse
   operators are in the energy eigenbasis: operators connecting neighbouring
   levels give far smaller errors than operators connecting distant ones.
   Measured locality at dimension 64 is 27%, 26% and 3.1% for the three systems
   below — see §2.5 for the definition and the caveats. This is an empirical
   correlation across three systems, not a derived law.

The three systems benchmarked here isolate these two conditions. In the
locality row, the percentage is the average energy-level separation a collapse
operator connects, as a fraction of the whole spectrum: **3% means an operator
essentially only couples neighbouring levels, 27% means it couples levels a
quarter of the spectrum apart.** The definition is in §2.5.

| Metric / Property | System A — Spin Chain ($g=0$, §2.3) | System B — Mixed Chain ($g=0.4$, §2.3) | System C — Oscillator (§2.4) |
|---|---|---|---|
| Model | Transverse-field Ising ($J=1, h=0.6$) | Mixed-field Ising ($g=0.4$) | Anharmonic oscillator + spin |
| Coupling $X$ | $\sum_i \sigma_x^i$ (collective) | $\sum_i \sigma_x^i$ (collective) | $x \otimes I$ (position) |
| $N_L$ at dim 64 | 31 (collapses due to integrability) | 2,017 ($\sim N^2 / 2$) | 890 ($\sim N^2$) |
| How far operators reach, $\bar d$ (dim 64) | 27% of the spectrum | 26% | **3.1% — neighbours only** |
| Cost versus the exact solve | **none** (1.0x at dim 64) | **547x cheaper** (dim 128) | **54x cheaper** (dim 64) |
| Relative error, one run at $M=16$ (dim 64) | $3.6\times10^{-2}$ | $5.4\times10^{-2}$ | $\mathbf{6.0\times10^{-6}}$ |
| Role here | **Control 1** — too few operators to bundle | **Control 2** — many operators, but each reaches far | **Demonstration** — many operators, each local |

**System A** is a special case. Its bath couples to every spin in the same way,
and the model itself is exactly solvable, which makes its energy gaps repeat
over and over. The Davies construction merges every transition that shares a
gap into a single operator, so even a 512-dimensional chain ends up with just
73 of them. Bundling works by shrinking a long list of operators; with a list
this short there is nothing to shrink, and solving exactly is the better choice
at every size measured.

**System B** is the same chain with one extra field term, which destroys that
special structure. The gaps stop repeating, almost nothing merges, and the
count climbs to 8,193 at dimension 128 — enough to make bundling **547x
cheaper** than solving exactly. But its operators connect energy levels far
apart rather than neighbouring ones, and a single run at $M=16$ lands at
percent-level error.

**System C** has both properties at once: many operators (890 at dimension 64)
*and* operators that only connect neighbouring rungs of its ladder. It is the
only system here that is cheap **and** accurate at small $M$.

### Will bundling help *your* system?

Two checks, both cheap, both before you run anything expensive.

**1. Count the collapse operators.**

```python
from qutip_bundling import davies_operators
n_l = len(davies_operators(H, X, gamma))
```

Bundling replaces $N_L$ operators with $M$, so the best speedup available is
about $N_L/M$. If $N_L$ is a few dozen, there is nothing to win — the exact
solve is cheaper and exact. If it is in the hundreds or thousands, the saving
is real. This part is arithmetic, not a claim: it holds by construction.

Be aware that $N_L$ is **not** predictable from the Hilbert dimension. Symmetry
and integrability can collapse it by orders of magnitude, as they do in System
A. Count it; do not estimate it.

**2. Ask how far each operator reaches.** Diagonalise $H$, write the collapse
operators in that basis, and look at how far from the diagonal their weight
sits ($\bar d$, §2.5). Operators confined near the diagonal — a ladder, a
cascade, anything where the bath moves the system one level at a time —
accompany the smallest errors here by four orders of magnitude. Operators
spread across the spectrum accompany percent-level error at the same $M$.

This second check is an **empirical correlation across three systems, not a
derived law**, and §2.5 says exactly where it fails. Treat it as a guide to
what to measure, not a guarantee.

**What to do with the answer.**

| $N_L$ | operator reach | what to expect |
|---|---|---|
| small (tens) | either | use the exact solver; bundling has nothing to compress |
| large | local | the best case — large speedup at small $M$ (System C) |
| large | far-reaching | still a large speedup, but you will need a larger $M$ for a given accuracy (System B) |

And if you cannot even build the operator list — thousands of dense operators
at large dimension — that is the regime bundling exists for, though you will
then have no exact reference to check against.

Both checks run on all three systems with one command, and the script is short
enough to copy and point at your own $(H, X)$:

```bash
python benchmarks/explain_structure.py
```

§2.6 walks through its output and explains where each number comes from.

There are also two stochastic methods on the table, and the single most
important thing to understand up front is that **they randomize different
things.** This is why their costs and errors behave so differently, and why the
comparison has to be empirical.

**`mcsolve` randomizes the state.** It *unravels* the master equation into
random pure-state trajectories. One trajectory is a wavefunction
$|\psi(t)\rangle$ that drifts under the non-Hermitian effective Hamiltonian
$H_{\rm eff} = H - \tfrac{i}{2}\sum_a L_a^\dagger L_a$, interrupted by random
*quantum jumps*: at random times one of the original $N_L$ collapse operators
$L_a$ fires (chosen with probability
$\propto\langle\psi|L_a^\dagger L_a|\psi\rangle$) and the state resets to
$L_a|\psi\rangle$. A single trajectory
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

## 2. The three test systems (fully specified)

### 2.0 The physical picture, before any equations

Every system here is the same kind of object: a **small quantum system** in
contact with a **large environment**, held at a fixed temperature.

**The system** is what we simulate — a row of spins, or a vibrating mode with a
spin attached. It is small enough to write down completely.

**The bath** is everything else: the phonons of a surrounding crystal, the
photons of a cavity, the solvent a molecule sits in. It is far too large to
simulate, and we never do. Instead it is *summed over* and replaced by its
effect on the system — draining energy away and destroying quantum coherence.
Its entire influence is captured by one function, $\gamma(\omega)$, giving how
readily the bath absorbs or emits an energy quantum $\omega$ (§2.1).

**The coupling operator $X$** is the part that decides *what the environment can
see*. A bath does not touch a system everywhere at once; it couples to one
physical property. For a molecule jostled by a solvent, that property is
position. For a spin in a fluctuating magnetic field, it is a spin component.
Choosing $X$ is choosing which handle the environment has.

**What "weak coupling" buys.** When the system–bath coupling is weak, the
environment can be eliminated exactly to leading order, leaving a closed
equation for the system alone — the Lindblad master equation. The price is a
set of **collapse operators**: each one is a channel through which the bath can
move the system between its energy levels.

#### How the collapse operators are built (the "construction")

Throughout this page, *construction* means the recipe that turns a Hamiltonian
and a coupling operator into that list of collapse operators. All three systems
use the same one, `davies_operators(H, X, gamma)`:

1. **Find the energy levels.** Diagonalise $H$.
2. **Look at every pair of levels.** A pair separated by energy $\omega$ is a
   possible transition, and $X$ says how strongly the bath drives it.
3. **Group transitions that share a gap.** This is the important step. The bath
   responds to a *frequency*, not to a pair of labels — if two different pairs
   of levels happen to be separated by the same $\omega$, the environment
   cannot tell them apart, so they are one channel, not two. All blocks with
   the same $\omega$ are summed into a single operator $A(\omega)$.
4. **Weight by the bath.** Multiply by $\sqrt{\gamma(\omega)}$: how fast the
   bath actually drives that frequency.

The number of collapse operators, $N_L$, is therefore **the number of distinct
energy gaps that the coupling operator can drive** — not the number of levels,
and not the dimension. That is why it varies so wildly between the systems
below, and it is the single quantity that decides whether bundling helps.

#### Two terms used below

**Exact symmetry.** An operation that leaves the Hamiltonian completely
unchanged. System A, for instance, is unchanged if you flip *every* spin about
the $x$ axis at once ($\mathbb{Z}_2$ symmetry). Symmetries have two
consequences here: they force some transitions to vanish and make others share
a gap, shrinking $N_L$; and they can split the state space into sectors the
dynamics never mixes, so the system remembers which sector it started in. That
is why detailed balance makes the thermal (Gibbs) state stationary, but does
not always make it the *unique* late-time state.

**Integrability, and "free fermions".** A model is integrable when it can be
transformed into something non-interacting. System A can: a standard change of
variables turns the interacting spin chain into a gas of independent
particles. Because the particles do not interact, the total energies are simply
*sums of n independent single-particle energies* — and sums built from the
same small pool of numbers produce the same differences over and over. Its gaps
repeat massively, step 3 above merges them, and $N_L$ collapses. Adding one
extra field term (System B) breaks the transformation, the gaps stop repeating,
and $N_L$ grows by orders of magnitude. Nothing else about the chain changes.

Both terms are demonstrated on the actual spectra in **§2.6** — including an
explicit check that System A's 16 energies are the subset sums of 4 numbers, to
machine precision, and System B's are not.

---

All three systems are weakly coupled to the **same** thermal bath, use the
**same** construction, and are scored on the **same** observables. They differ
only in $(H_{\rm sys}, X)$ — which makes them a controlled comparison rather
than three unrelated demonstrations. §2.5 states what each one is for.

### 2.1 The bath (shared by all three systems)

The bath is specified entirely by one spectral function — the rate at which the
bath exchanges energy quantum $\omega$ with the system:

$$
\gamma(\omega) =
\frac{\alpha\,\omega\,e^{-|\omega|/\omega_c}}{1-e^{-\omega/(k_BT)}},
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
  $\gamma(-\omega)/\gamma(\omega)=e^{-\omega/k_BT}$), which makes the Gibbs
  state stationary; uniqueness additionally requires the coupling to connect all
  symmetry sectors. At $\omega\to0$
  this factor gives the finite limit $\gamma(0)=\alpha\,k_BT$.

So in one line: **an ohmic bath with an exponential cutoff, at temperature
$k_BT=0.5$, satisfying detailed balance.**

**Building the Lindblad operators.** Both systems turn
$(H_{\rm sys}, X, \gamma)$ into collapse operators by the same strict Davies
(secular) recipe. First resolve the Hamiltonian into projectors onto its
distinct, possibly degenerate, energy eigenspaces,

$$H_{\rm sys}=\sum_\epsilon \epsilon\,\Pi_\epsilon.$$

Then group every transition block with the same Bohr frequency:

$$
\omega=\epsilon'-\epsilon,\qquad
A(\omega)=\sum_{\epsilon'-\epsilon=\omega}
\Pi_\epsilon X\Pi_{\epsilon'},\qquad
c(\omega)=\sqrt{\gamma(\omega)}\,A(\omega).
$$

There is one collapse operator per **distinct populated frequency sector**, not
one per eigenvector pair. Keeping the sum inside $A(\omega)$ retains the cross
terms required when energies or gaps are degenerate and makes the generator
independent of the arbitrary eigenbasis chosen inside a degenerate eigenspace.
The code groups numerical equalities within
`DAVIES_DEGENERACY_TOL = 1e-10`. It also removes projector blocks at or below
the scale-covariant backward-error floor
$512\,\epsilon_{\rm mach}N\lVert X\rVert_F$, so exact symmetry zeros do not
become machine-dependent operators.

**Why that value, and how to choose your own.** The tolerance is a genuine
physical knob — it decides which Bohr frequencies count as one bath channel, so
it sets $N_L$ — and it should be chosen so that the answer does not depend on
it. Sweeping it across nine decades on the benchmark systems gives a wide
plateau:

| tolerance | $10^{-14}$ | $10^{-12}$ | $10^{-10}$ | $10^{-8}$ | $10^{-6}$ | $10^{-5}$ | $10^{-4}$ | $10^{-3}$ |
|---|---|---|---|---|---|---|---|---|
| System B, dim 64 | 2,017 | 2,017 | **2,017** | 2,017 | 2,017 | 2,015 | 1,991 | 1,813 |
| System C, dim 64 | 890 | 890 | **890** | 890 | 890 | 890 | 890 | 884 |

Everything from $10^{-14}$ to $10^{-6}$ gives an identical count; erosion starts
at $10^{-5}$ and becomes severe past $10^{-4}$. $10^{-10}$ sits at the middle of
that shelf, four orders below the smallest separation between two genuinely
distinct driven frequencies ($2.5\times10^{-6}$ for System B at dimension 64).

Two things follow for anyone applying this elsewhere. **A tolerance that changes
your count is too loose** — find the plateau first. And **the plateau existing
at all is the evidence that eigenvalue roundoff is not inflating $N_L$**: if it
were, tightening below $10^{-10}$ would split operators, and it does not.
`tests/test_degeneracy_tolerance.py` pins both properties.

The sign $\omega=\epsilon'-\epsilon$ makes a downward transition positive.
With the detailed-balance convention above, the Gibbs state is stationary.
It is the unique late-time state only when $(H_{\rm sys},X)$ is ergodic; an
exact symmetry shared by both can preserve multiple stationary sectors.

The three systems feed *different* $(H_{\rm sys}, X)$ into this one recipe:

- **System A** (§2.3): Integrable Ising chain ($g=0$), $X = \sum_i \sigma^x_i$. Its free-fermion integrability and $\mathbb{Z}_2$ symmetry collapse the Bohr spectrum. At 6 spins (dim 64), $N_L = 31$ ($N_L = n^2-n+1$).
- **System B** (§2.3): Mixed-field Ising chain ($g=0.4$), $X = \sum_i \sigma^x_i$. The longitudinal field breaks integrability, preventing frequency collapse and raising $N_L$ to 2,017 at dim 64 and 8,193 at dim 128. Its transition matrices in the energy eigenbasis are dense (bandwidth ~16% under the `probe_oq4_accuracy.py` normalisation; ~26-32% under the mean-distance definition given in §1 -- quote which one you mean).
- **System C** (§2.4): Anharmonic oscillator + spin, $X = x \otimes I$. Its collapse operators act through position $x \propto (a + a^\dagger)$, which is strictly tridiagonal in Fock space. $N_L = 890$ at dim 64, and its transition bandwidth is narrow (~3.1% – 4.4% of the spectrum).

In code these are built via dedicated functions in `common.py`:

```python
H, X, psi0 = build_spin_chain(6, g=0.0)      # System A  (dim 64, N_L=31)
H, X, psi0 = build_spin_chain(6, g=0.4)      # System B  (dim 64, N_L=2017)
H, X, psi0 = build_oscillator_bath(64)       # System C  (dim 64, N_L=890)
c_ops = davies_operators(H, X, gamma)        # the collapse operators, length N_L
```

### 2.2 Is this weak coupling? Yes — in both senses.

1. **By construction.** `davies_operators` builds a Davies/secular master
   equation, which is *derived* in the weak system–bath coupling (Born–Markov)
   limit. Using the Davies operators places the model in the weak-coupling
   Lindblad regime by assumption — that is the theory's domain of validity.
2. **By the numbers.** The bath coupling scale $\alpha=0.3$ is smaller than each
   system's coherent energy scales ($J=1$ for the chains, $\omega_0=1$ for the
   oscillator), so dissipation is slower than the internal coherent dynamics —
   the weak-coupling ordering. It is "moderate" weak coupling: strong enough to
   produce real relaxation over $t\in[0,5]$, not so strong that the perturbative
   description breaks.

### 2.3 System A & B — transverse-field ($g=0$) and mixed-field ($g=0.4$) Ising chains

**Physically:** a row of $n$ tiny magnets. Neighbours prefer to point the same
way (the $ZZ$ term), while a magnetic field along $x$ tries to tip them all
sideways (the $X$ term). System B adds a second field along $z$.

The bath couples through $X=\sum_i\sigma^x_i$ — the environment pushes on
*every* spin identically, along $x$. It cannot address one spin: picture the
whole chain sitting in a single fluctuating field, rather than each magnet
having its own noise source. The chain starts fully polarised and relaxes
towards thermal equilibrium.

The **system Hamiltonian** for $n$ spins is

$$
H_{\rm sys} = -J\sum_{i=1}^{n-1}\sigma^z_i\sigma^z_{i+1}
              - h\sum_{i=1}^{n}\sigma^x_i
              - g\sum_{i=1}^{n}\sigma^z_i,
\qquad J = 1.0,\; h = 0.6 .
$$

- **System A ($g=0.0$):** The transverse-field Ising model is **integrable**. It maps to free fermions, so its energy levels are sums of $n$ independent single-particle mode energies. Enormous numbers of transitions share identical Bohr frequencies, collapsing $N_L$ under Davies grouping to $n^2-n+1$ (31 at dim 64, 43 at dim 128).
- **System B ($g=0.4$):** Adding the longitudinal field $g=0.4$ **breaks integrability**. Bohr frequencies no longer coincide, grouping merges almost nothing, and $N_L$ scales as $\sim N^2 / 2$ (2,017 at dim 64, 8,193 at dim 128).

In both chains, the bath couples through $X = \sum_i \sigma^x_i$, and the chain starts fully polarized, $|\psi_0\rangle = |{\uparrow\uparrow\cdots\uparrow}\rangle$. Expressed in the energy eigenbasis, $X$ produces **dense** transitions: it
connects levels far apart in energy rather than only neighbouring ones. Under
the mean-transition-distance definition of §1 this is ~26-32% of the spectrum
for both chains, essentially independent of $g$ — breaking integrability
multiplies the operator *count* without making the individual operators any
more local.

![System A schematic](system_a_schematic.png)

System A / B: an Ising chain ($J = 1.0$, $h = 0.6$, $g=0$ or $g=0.4$), fully
polarized at $t = 0$, coupled through the global operator
$X = \sum_i \sigma^x_i$ to a single collective ohmic bath.

### 2.4 System C — anharmonic oscillator coupled to a spin

**Physically:** a single vibrating mode — think of one bond in a molecule —
with a two-level system attached. The vibration's energy ladder is
*anharmonic*: its rungs are not evenly spaced but widen going up, much as a
real bond stiffens as it stretches.

The bath couples through the oscillator's **position**, $X = x\otimes I$: a
surrounding solvent or lattice jostles the bond and damps its motion. Two
consequences matter for everything below. First, position only connects
*adjacent* rungs of the ladder ($n\to n\pm1$), so the environment can only walk
the oscillator down one step at a time — this is the "local" transition
structure of §1. Second, the bath never touches the spin at all; the spin feels
dissipation only second-hand, through its coherent coupling to the oscillator.

The oscillator starts at the top of its ladder and cascades down.

The **system Hamiltonian** is

$$
H_{\rm sys} = \omega_0\left(n+\tfrac12\right) + \chi n^2
              + \tfrac{\Delta}{2}\sigma_z + g_{\rm int}(x\otimes\sigma_x)
$$

with $\omega_0=1.0$, anharmonicity $\chi=0.1$, spin gap $\Delta=1.0$, and an
internal oscillator–spin coupling $g_{\rm int}=0.3$ (`coupling` in the code;
written $g_{\rm int}$ here because $g$ already denotes the chains'
longitudinal field in §2.3 — they are unrelated). Here $n=a^\dagger a$ is the number
operator and $x=(a+a^\dagger)/\sqrt2$ the position. The four terms are: the bare
oscillator, its anharmonicity, the spin's energy splitting, and a coherent
oscillator–spin coupling.

![System C schematic](system_b_schematic.png)

System C: an anharmonic oscillator whose energy gaps widen up the ladder,
coupled to a two-level spin by an internal coherent coupling $g_{\rm int}$. A single
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
coherent coupling $g_{\rm int}(x\otimes\sigma_x)$. The system starts in the oscillator's
top Fock state with the spin down. As above, the total evolved object is the Lindblad master
equation with dissipators built from $(H_{\rm sys}, X, \gamma)$ as in §2.1 ($N_L = 128$ at dim 16). Note the two
distinct "couplings": $g_{\rm int}=0.3$ is an **internal coherent** coupling inside
$H_{\rm sys}$, whereas $\alpha=0.3$ is the **system–bath** coupling carried by
$\gamma$ through $X$ — they are different physics that happen to share a value.
The size is set by the Fock truncation. This system is close to the
molecular/vibronic problems the method was developed for.

### 2.5 What each system is for

The three are not three demonstrations. They vary two properties independently,
so the benchmark can say which one matters:

| | operator count $N_L$ | how far each operator reaches | outcome |
|---|---|---|---|
| **A** — TFIM chain ($g=0$) | small (31 at dim 64) | reaches far ($\bar d\approx27\%$) | no speedup available |
| **B** — mixed chain ($g=0.4$) | large (2,017 at dim 64) | reaches far ($\bar d\approx26\%$) | 547x cheaper, error $5.4\times10^{-2}$ |
| **C** — oscillator | large (890 at dim 64) | local ($\bar d\approx3\%$) | 54x cheaper, error $6.0\times10^{-6}$ |

**A versus B** isolates the operator count. Same lattice, same coupling
operator, same initial state; the only change is a longitudinal field that
breaks integrability. `N_L` rises 65-fold and a speedup appears where there was
none — so the *cost* benefit is governed by `N_L`, as §1 claims.

**B versus C** isolates the operator structure. Both have thousands of
operators, so both are cheap. But B's transitions are dense in energy and C's
are tridiagonal, and their accuracies differ by four orders of magnitude — so
the *accuracy* is not governed by `N_L`. Without B, that would be invisible,
and the natural reading of A-versus-C would be that a large `N_L` buys both
benefits at once. It does not.

**System B is therefore not a failure.** A 547x speedup at ~5% error is a
useful operating point — for parameter sweeps, screening, or qualitative
dynamics — and it is a *different point on the cost-accuracy curve*, not a
broken result. What it rules out is the simpler claim we would otherwise have
made.

**Measuring locality.** Order the eigenvectors $|i\rangle$ of $H$ by energy and
take the mean transition distance, in units of the dimension:

$$
\bar{d} = \frac{1}{N} \frac{\sum_{\alpha}\sum_{ij} |i-j| \, |\langle i|c_\alpha|j\rangle|^2}{\sum_{\alpha}\sum_{ij} |\langle i|c_\alpha|j\rangle|^2}
$$

At dimension 64 this gives 27% (A), 26% (B) and 3.1% (C). `probe_oq4_accuracy.py`
normalises differently, so state which definition you mean — the ranking is
robust, the absolute scale is not.

**Treat it as a guide, not a law.** It separates the two *families* cleanly, and
within System C halving $\bar{d}$ divides the error by about five. But it does
not explain everything: within the chains $\bar{d}$ falls with dimension while
the error rises, and System C's fitted power law under-predicts the chains'
error by a factor of 40. It also sets a prefactor rather than a floor — System B
still reaches $2.4\times10^{-3}$, it just needs $M=128$ to get there.

A reader with their own model can place it on this table by computing two
numbers before running anything: `len(davies_operators(H, X, gamma))` and the
mean transition distance defined in §1. `explain_structure.py` computes both,
and §2.6 walks through what they mean on these three systems.

### 2.6 Worked example: Why do some systems compress so well?

In the tables above, you might notice that **System A** (the simple spin chain) ends up with a remarkably small number of Lindblad operators ($N_L = 13$ at dimension 16), while **System B** (the mixed chain) and **System C** (the oscillator) do not. 

Why does the Davies method compress System A so effectively, but leave the others mostly alone? The script `explain_structure.py` breaks this down. There are two physical reasons:

#### 1. The Energy Gaps "Collide" (Degeneracy)
The Davies method groups transitions together if they share the *exact same energy gap* (the difference in energy between the starting state and the ending state). 

Imagine a system where the energy levels are completely messy. If you calculate the gap between any two levels, you'll get a unique number every time. No two gaps will be identical, so no grouping can happen. This is exactly what happens in System B and System C.

System A is special. It is a "free" model, meaning its excitations (like flipping a spin) don't interfere with each other. If flipping one spin costs 2 units of energy, and flipping another costs 3 units, then flipping *both* costs exactly 5 units. Because the energies add together perfectly, the energy gaps between different levels start to repeat over and over again. 

For a 4-spin chain (16 energy levels), there are 256 possible transitions. But because the energies add up perfectly, there are only 81 unique energy gaps. The gaps "collide," allowing the Davies method to pack many transitions into a single operator.

#### 2. The "Parity" Rule (Symmetry)
System A has a perfect symmetry: if you flip all the spins at once, the system's physics remain exactly the same. Because of this, every quantum state in the system is stamped with a built-in "parity" (think of it as being either "Even" or "Odd").

The coupling operator that connects the system to the bath ($X$) obeys this symmetry too. The rules of quantum mechanics dictate that $X$ is completely forbidden from causing a jump between an "Even" state and an "Odd" state. 

This simple rule instantly crosses out 76% of all the possible transitions on the board, before we even start grouping them by energy.

#### The Final Breakdown (Dimension 16)

Here is how those two rules play out in practice:

| | Allowed Transitions | Unique Energy Gaps ($N_L$) | Transitions packed into one operator |
|---|---|---|---|
| **System A** (Perfect symmetry, perfect adding) | 62 of 256 (24%) | **13** | 4.8 |
| **System B** (Broken symmetry, messy adding) | 136 of 256 (53%) | **121** | 1.1 |
| **System C** (No symmetry, messy adding) | 128 of 256 (50%) | **128** | 1.0 |

As you can see, System A gets a massive discount because of its symmetrical and "free" nature. Systems B and C do not have these clean physical properties, so their transitions don't group together. 

This proves a key point: **Extreme operator compression is a lucky feature of specific, clean physical models (like System A), not a guarantee for all systems.**

#### What the collapse operators look like

To see *why* System C works and System B fails, we can plot the total transition weight (how strongly the bath moves population between energy levels) as a heatmap over the 16x16 energy basis:

![System B Matrix](matrix_system_b.png)
![System C Matrix](matrix_system_c.png)

**System C is a staircase.** The bath couples through position
$x \propto a + a^\dagger$, which by construction has matrix elements only
between adjacent Fock states, $n \to n\pm1$. Anharmonicity mixes the states a
little, but the ladder survives: the weight sits on a narrow band hugging the
diagonal, and the bath drains the oscillator **one rung at a time**. The band
sits two indices out rather than one only because each oscillator level carries
the spin's two states with it — the physical step is still one rung.

**System B is scattered.** $X = \sum_i \sigma^x_i$ is local *on the lattice*,
but the energy eigenstates of a non-integrable chain are superpositions spread
over the whole chain, so in the energy basis a single spin flip connects levels
far apart in energy. One bath event can take the system from near the bottom of
the spectrum to near the middle. There is no ladder to walk.

$\bar d$ puts one number on that difference: the strength-weighted mean of
$|a-b|$. C sits at 1.97 out of a possible 15; B at 5.13, close to the 5.3 you
would get by scattering weight uniformly at random.

Note what this does **not** separate. System A's $\bar d$ is 4.98 — statistically
the same as B's. Locality does not distinguish A from B; $N_L$ does (13 against
121). And $N_L$ does not distinguish B from C (121 against 128); locality does.
That is precisely why three systems are needed and two would not do.

#### Where the error actually comes from

It is tempting to think locality works because sampling noise "cancels out." It doesn't. 

Write out one bundled dissipator $R = M^{-1/2}\sum_\alpha r_\alpha c_\alpha$ with random signs $r_\alpha$:

$$
R \rho R^\dagger = \frac{1}{M}\sum_{\alpha}c_\alpha \rho c_\alpha^\dagger + \frac{1}{M}\sum_{\alpha\neq\beta} r_\alpha r_\beta \, c_\alpha \rho c_\beta^\dagger
$$

The first sum is the exact dissipator. **The second sum is the entire error.** Its average is zero, but at finite $M$ it survives. 

The question isn't "how local are the operators," but **how many cross-terms survive to create noise?**

| | non-vanishing cross terms | total cross weight ÷ true dissipator |
|---|---|---|
| **A** | 100% | 1.9x |
| **B** | 56% | **4.2x** |
| **C** | 11% | **0.6x** |

**Locality is simply *why* C's cross terms vanish.** Because C is a ladder, taking a step down ($c_\alpha$) and a step back up ($c_\beta^\dagger$) rarely lines up, so 89% of the products evaluate to exactly zero. System B is scattered, so almost every path survives.

**Is this ratio a better predictor than $\bar d$?**
Yes, but it is still not a universal law. 
- **It fixes the direction:** Within each system, a higher ratio correctly predicts higher error (which $\bar d$ failed to do).
- **It fails across systems:** It massively under-predicts the gap between the oscillator and the chains.
- **It gets A vs B backwards:** At dimension 64, System A has a much smaller ratio than B, but larger error. 

The cross-term ratio is the exact algebraic origin of the error, but it still cannot quantitatively predict error across different systems. **Open Question 4 stays open.** 

**A note on the rate.** The sweeps consistently confirm that the empirical error falls as $1/M$, not $1/\sqrt{M}$: doubling $M$ halves it, measured at 1.87x-2.04x per doubling across every system and dimension.



---

## 3. What we measure, and how the error is reported

### 3.1 Which observables, and why not just the energy

An **observable** is anything you can measure from the state. Pick an operator
$A$; at each time it gives one number, $\langle A\rangle(t)=\mathrm{Tr}(A\rho(t))$,
and over the run that traces a curve. Every accuracy figure on this page is a
comparison of two such curves — one from the method being tested, one from the
exact reference.

Which quantities you pick is not a side detail. **It decides what "accurate"
means.**

#### Why the energy alone is not enough

The state $\rho$ is a matrix, and its two halves mean different things:

- the **diagonal**, $\rho_{aa}$ — how much of the state sits in energy level
  $a$. These are the *populations*.
- the **off-diagonal**, $\rho_{ab}$ — how strongly levels $a$ and $b$ are still
  in step with each other. These are the *coherences*. They have no classical
  counterpart, and they are the first thing an environment destroys.

$\langle H\rangle$ is built almost entirely from the diagonal. So a method can
get the populations right, get the coherences badly wrong, and the energy curve
will still look fine.

That is not a small effect here. On identical runs, SLB beats `mcsolve` by
**488x on the energy** but only **7x on the dominant coherence**. Quote the
energy alone and you overstate the advantage roughly seventyfold.

**Read every accuracy claim on this page as being about one specific
observable.**

#### How the observables were chosen

The Hamiltonian is a sum of terms, $H=\sum_k \lambda_k O_k$. Each term $O_k$ is
measured on its own. Two things come for free:

1. **They have to add back up to the energy.** The identity
   $\langle H\rangle=\sum_k\lambda_k\langle O_k\rangle$ is checked on every run and holds to machine precision —
   residual $9\times10^{-16}$ on the chains, $2\times10^{-15}$ on the
   oscillator. That is a correctness test on the whole pipeline, at no extra
   cost.
2. **Each one already has a name.** $O_k$ is the susceptibility
   $\partial\langle H\rangle/\partial\lambda_k$, so these are quantities people
   already quote for these models, not probes invented for this benchmark.

| | chains (A, B) | oscillator (C) |
|---|---|---|
| the bulk number | $\langle H\rangle$ | $\langle H\rangle$ |
| the quantum part | dominant coherence | dominant coherence |
| what the bath drives | $\sum_i\langle\sigma^z_i\sigma^z_{i+1}\rangle$ — neighbour alignment | $\langle n\rangle$ — how far up the ladder |
| the hard one | $\sum_i\langle\sigma^x_i\rangle$ — this *is* the bath coupling operator | $\langle\sigma^z\rangle$ — the bath never touches the spin |
| remaining $H$ terms | $\sum_i\langle\sigma^z_i\rangle$ (System B only) | $\langle n^2\rangle$, $\langle x\otimes\sigma^x\rangle$ |
| per-site version | $\sum\langle\sigma^z\sigma^z\rangle/(n-1)$ | — |

#### What each observable tells you

**$\langle H\rangle$ — total energy.** The weighted sum of every other observable.
It is dominated by the diagonal of $\rho$ (the populations), so a method can get
the off-diagonal structure badly wrong and the energy curve will still look
fine. It is necessary but not sufficient.

**Dominant coherence — quantum interference between the two most-coupled energy
eigenstates.** The density matrix $\rho$ is rotated into the eigenbasis of $H$,
so its off-diagonal entries $\rho_{ab}$ measure how much quantum superposition
exists between energy levels $a$ and $b$. Most pairs are effectively dead (at
dim 16, 76 out of 120 never reach 1 % of the largest). The code
(`common.populated_coherence_op`) scans the exact reference trajectory and picks
the single pair $(a,b)$ whose $|\rho_{ab}|$ is largest at any time. The
observable is $|a\rangle\langle b|+|b\rangle\langle a|$, so it traces a
real-valued curve. This directly tests whether the method preserves quantum
interference, not just level populations.

**$\sum_i\langle\sigma^z_i\sigma^z_{i+1}\rangle$ — nearest-neighbour Ising
correlation (chains).** This is the coupling between adjacent spins in the
$z$-direction. It tells you whether the chain is ordering ferromagnetically
(spins aligning) or remaining disordered under the bath. Since the bath drives
transitions through this term, getting it right means the dissipation is being
approximated correctly.

**$\langle n\rangle$ — oscillator excitation number (System C).** The average
rung on the harmonic-oscillator ladder. If the bath is thermalizing the
oscillator, $\langle n\rangle$ should relax toward the Bose–Einstein value set
by the bath temperature. This directly tests whether the bath is doing its job.

**$\sum_i\langle\sigma^x_i\rangle$ — bath coupling operator (chains).** The bath
drives transitions *through* $\sigma^x$, so this expectation value is the most
sensitive to how accurately you approximate the dissipation. Any error in the
bundled collapse operators shows up here first, before it bleeds into the other
observables. That is why it is labelled "the hard one."

**$\langle\sigma^z\rangle$ — the spin the bath never touches (System C).** The
bath couples only through $X=x\otimes I$, so it never acts on the spin at all.
The spin moves *solely* because it is coupled to the oscillator internally. On
top of that it carries under 4 % of the total energy, so it can be badly wrong
while $\langle H\rangle$ barely flinches. That blind spot is exactly why it is
included.

**$\sum_i\langle\sigma^z_i\rangle$ — longitudinal magnetization (System B
only).** The mixed-field Ising chain adds a $\sigma^z$ field on top of the
transverse $\sigma^x$ field. This observable tracks the response to that extra
field and is absent in the pure transverse-field chain (System A).

**$\langle n^2\rangle$ — oscillator excitation variance (System C).** The second
moment of the oscillator number. Together with $\langle n\rangle$ it pins down
the width of the excitation distribution — a check that energy alone cannot
provide, since two very different distributions can share the same
$\langle n\rangle$.

**$\langle x\otimes\sigma^x\rangle$ — oscillator–spin correlation (System C).**
This is the internal coupling term between the oscillator and the spin. It
measures how strongly the two subsystems are entangled or correlated. A method
that treats the two subsystems too independently will get this wrong even if it
nails the marginals $\langle n\rangle$ and $\langle\sigma^z\rangle$ separately.

**$\sum\langle\sigma^z\sigma^z\rangle/(n{-}1)$ — per-bond correlation
(chains).** The same nearest-neighbour correlation divided by the number of
bonds. It rescales the chain observable so that different chain lengths can be
compared on the same plot.

#### The coherence is chosen from evidence, not by hand

At dimension 16 there are 120 possible level pairs $(a,b)$, and only one of them
becomes the coherence observable. The choice matters enormously, because **most
pairs are dead**. Running the exact dynamics and recording the largest
$|\rho_{ab}|$ each pair ever reaches:

| pair | largest $|\rho_{ab}|$ over the whole run |
|---|---|
| $(0,1)$ | $3.9\times10^{-1}$ |
| $(1,3)$ | $2.2\times10^{-1}$ |
| $(0,3)$ | $2.0\times10^{-1}$ |
| … | |
| $(10,14)$ | $2.9\times10^{-18}$ |

Top to bottom spans **17 orders of magnitude**, and **76 of the 120 pairs never
reach even 1%** of the largest. The dynamics simply never creates coherence
between those levels.

Pick one of those by hand and the exact answer is zero, SLB returns zero,
`mcsolve` returns zero, and a broken method returns zero too. Every method
passes and nothing has been measured.

So `common.populated_coherence_op` picks by evidence instead: it scans the
reference trajectory, rotates $\rho$ into the energy eigenbasis, blanks the
diagonal, and takes the pair whose $|\rho_{ab}|$ is largest at any time. The
observable is $|a\rangle\langle b|+\mathrm{h.c.}$

**The pair is fixed once, from the reference, and handed to every method.** If
each method found its own largest pair, SLB would report its error on one pair
and `mcsolve` on another — two numbers about two different quantities, which
cannot be put side by side in a table.

#### One limit worth knowing if you extend this

Trajectory methods rebuild $\rho$ as an *average over samples*, so they can only
report quantities that are **linear in $\rho$**. Purity $\mathrm{Tr}(\rho^2)$
and the von Neumann and entanglement entropies are not: averaging
$\mathrm{Tr}(\rho_k^2)$ over trajectories does not give $\mathrm{Tr}(\bar\rho^2)$.
Density-matrix methods, bundling included, carry no such restriction, and that
is a genuine advantage of that route.

Two smaller rules. Observables must be Hermitian, or the solvers quietly discard
an imaginary part. And they must be declared *before* the run, because `mcsolve`
cannot store full states at these sizes and has to be told up front what to
record.

### 3.2 Error: a time-resolved band, and the single numbers from it

The quantity of interest is how well the bundled $\langle H(t)\rangle$ (and, in
Result 1, a coherence) tracks the exact reference. The accuracy and coherence
figures show this **resolved over the whole trajectory**: for each $M$, the SLB
mean curve is drawn against the reference with a shaded **$\pm1$
standard-deviation band** (the spread over stochastic realizations). The
separate error-decomposition figure in Result 1 then resolves the bias and
fluctuation at the hardest instant.

This keeps the **two error components** separate:

- the **bias** is the offset of the bundled mean from the exact answer;
- the **statistical fluctuation** is the width of the shaded band — how much a
  single bundled run scatters around that mean.

The distinction matters because one combined number hides which dominates, and
at small $M$ or small systems the fluctuation can be *larger* than the bias.

When the error must instead live on an axis — the **error-vs-X figures** — it is
collapsed to one scalar per point, and the right scalar depends on the figure's
job:

- **Characterizing one method's own scaling** — convergence vs $M$ and the
  jackknife vs dimension (both §6) — uses the **max-over-time** error, the
  worst deviation over the trajectory. For "how fast does this method's error
  shrink," the conservative worst case is the robust diagnostic and needs no
  chosen time.
- **Comparing two methods head-to-head** — the SLB-vs-`mcsolve` frontier
  (Result 3) — uses the **time-averaged RMSE**: at each time it combines the
  systematic bias $|\langle H\rangle_{\rm SLB}-\langle H\rangle_{\rm ref}|$
  with the statistical error $S/\sqrt{N_r}$ as $\sqrt{\text{bias}^2+\text{SEM}^2}$,
  then averages over the trajectory. This is the fair choice for a head-to-head:
  it counts *both* error components — a bias-only or single-time number would
  ignore `mcsolve`'s large trajectory variance, and could be gamed by trading
  bias for variance or the reverse — and time-averaging avoids both a
  lucky single instant and the upward bias of a max-over-time number. The substep
  integrator check (§6) still reports a single **mid-relaxation time
  $t=2.5$**, where one representative instant suffices.

(The dynamics run to $t=5$ in natural units — $J=1$ for the chain,
$\omega_0=1$ for the oscillator. Result 2 and the validation checks use 40
output points; the accuracy-style Results 1, 3, and 4 use 80. In either grid,
$t=2.5$ is the mid-relaxation sample, where the energy has substantially
decayed but not yet saturated.)

### 3.3 How much sampling each method does

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

| figure | `M` | `n_realizations` | error bars |
|---|---|---|---|
| accuracy (Result 1) | system-dependent | 200 | $\pm1$ std band |
| cost scaling (Result 2) | 8 (iso-accuracy sweeps `M`) | 1 (cost) / 16 (RMSE) | — |
| frontier (Result 3) | 1–64, system/size-dependent | spin: 2 / 4 / 8; oscillator: 8 / 16 | $S/\sqrt{N_r}$ |
| iso-cost vs dim (Result 4) | swept to target ($\le 128$) | spin: 4; oscillator: 16 | mcsolve via $S/\sqrt{\texttt{ntraj}}$ fit |

**`mcsolve` has one level of sampling:** a single reported point is `ntraj`
independent trajectories (swept over `[10, 50, 200, 1000]` in the frontier
(Result 3), and sampled at `[100, 200, 400]` to fit the cost projection in
Result 4), run single-threaded so its wall-clock
time is the full sequential cost of all trajectories — matching SLB's
single-threaded realization loop. Its frontier error bar is its own trajectory
spread $S/\sqrt{\texttt{ntraj}}$ — the same quantity SLB's bar measures over its
runs, so the two methods are treated identically, one estimate per point (no
extra repeats of one method but not the other).

### 3.4 Integrators: matched where it is possible, disclosed where it is not

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
  **§6** makes this explicit) — so it does not contaminate the comparison.
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
nonlinear evolution leaves a **finite $M$ bias** of order $1/M$ in the state.
On top of that sits a run-to-run **fluctuation** that averages down over
realizations. So:

- **Bias** $\sim M^{-1}$ — reduced by raising `M` (or removed to leading order by
  the built-in jackknife correction). This is what grows with system size at
  fixed `M` (more operators compressed into the same bundles — see Result 2).
- **Fluctuation** — reduced by raising `n_realizations` (and also falls with
  `M`).

The practical upshot: `mcsolve` users turn one knob (`ntraj`) against noise; SLB
users turn `M` against bias and `n_realizations` against fluctuation. The
frontier (Result 3) sweeps the bias knob `M` for SLB against the noise knob
`ntraj` for `mcsolve`.

---

## 5. Results

> **Read in order, the benchmark results build one argument:**
> 1. **Memory and Stiffness Walls (§5.2):** `mesolve` hits a hard 32 GB memory wall at dim 128 for the chain and dim 64 for the oscillator; the oscillator hits a fixed-step RK4 stiffness ceiling at dim 256.
> 2. **Results 1 and 2:** accuracy versus bundle size, and cost scaling with dimension, regenerated under 0.6.4 for all three systems. On the mixed chain SLB at matched accuracy scales as $N^{1.8}$ against $N^{2.9}$ for the exact solver and $N^{4.4}$ for `mesolve`, so the advantage widens with size.
> 3. **Result 3 — the four-method comparison:** across three systems and six observables, SLB, `mcsolve`, and the exact solvers are compared head-to-head at dim 64. The advantage swings from 620x (oscillator energy) to 34x *worse* than `mcsolve` (mixed chain coherence). No single number captures the method; the section presents the full range.
> 4. **Result 4:** iso-accuracy cost versus dimension.

### 5.1 Reading the cost–accuracy plots

**What one point on the cost axis means.** Every wall-clock here is the *total*
for that method's whole ensemble on a single core: all 500 `mcsolve`
trajectories, all 16 SLB realizations, one deterministic solve. Both stochastic
methods parallelize trivially over their samples, so the figures carry a second
panel giving wall-clock divided by sample count — the limit of one core per
sample. The exact solvers have one sample and do not move between panels.

**The two stochastic methods need very different sample counts, and this is the
whole comparison.** `mcsolve`'s error is Monte-Carlo fluctuation, falling as
$1/\sqrt{N_{\rm traj}}$: at the oscillator, dimension 64, one trajectory gives
$6.7\times10^{-2}$ against $2.9\times10^{-3}$ for all 500 — a factor of 23,
against $\sqrt{500}=22$. It genuinely needs every trajectory. SLB's error at
fixed $M$ is dominated instead by the $O(1/M)$ *bias*, which averaging cannot
remove, so one realization lands within 1.3x of sixteen
($6.0\times10^{-6}$ against $4.7\times10^{-6}$). One realization is already the
answer.

| oscillator, dim 64 | samples it needs | cost of that | error |
|---|---|---|---|
| SLB, $M=16$ | **1** realization | 2.3 s | $6.0\times10^{-6}$ |
| `mcsolve` | **500** trajectories | 7,118 s serial, 14.2 s on 500 cores | $2.9\times10^{-3}$ |

So parallelism does not close the gap. Given unlimited cores `mcsolve` finishes
in 14.2 s at $2.9\times10^{-3}$, while SLB finishes in 2.3 s at
$6.0\times10^{-6}$ — still 6x faster and 490x more accurate. You cannot
parallelize away a $1/\sqrt{N}$ error; you can only pay for it.

**On the choice of exact baseline.** The speedups below are quoted against the
package's own native RK4 with the full dissipator, not against `mesolve`. That
matters: the two exact solvers disagree by up to two orders of magnitude on the
same problem — 219.6 s against 0.99 s on the TFIM chain at dimension 64, 179.3 s
against 23.2 s on the oscillator at dimension 32 — because `mesolve` builds the
full $N^2\times N^2$ Liouvillian. Quoting a speedup against `mesolve` would
partly measure that construction rather than bundling. Native RK4 shares SLB's
integrator and code path, so the only difference between them is $N_L \to M$,
which is the quantity under test.

### 5.2 Memory and Stiffness Walls

The solvers encounter two distinct, physical walls:

1. **The `mesolve` Memory Wall (32 GB):**
   - **Dimension Wall (Chain):** At dim 128, the chain's dense Liouvillian requires $> 32\text{ GB}$, triggering Out-Of-Memory (OOM).
   - **Operator Count Wall (Oscillator):** At dim 64, the oscillator's Liouvillian matrix is small (268 MB), but summing 890 superoperator matrices during construction exhausts 32 GB RAM.
2. **The Oscillator Stiffness Ceiling:**
   - The anharmonicity $\chi n^2$ grows with the Fock cutoff, and the substeps needed for
     stability roughly double per dimension doubling: 32 suffices at dim 64, 64 at dim 128,
     128 at dim 256.
   - **The dim-256 half of that rule has now been measured, and it holds.** It was a
     projection from two octaves; job 19592849 tested the third. Given 128 substeps SLB
     runs at dimension 256 -- one $M=8$ solve in 806.6 s, $N_L = 2{,}986$, Davies
     construction 19.4 s -- with no divergence. So the `slb_unstable_at_substeps: 32`
     recorded against that dimension in Result 2 means exactly what it says, *unstable at
     32*, and not that the method stops there.
   - **That point is deliberately absent from Result 2's curve**, for two independent
     reasons: it is integrated at $4\times$ that panel's substeps, and it ran on landau44
     while the panel ran on landau42. Either one alone disqualifies its wall-clock from
     that axis. Putting it on properly would mean re-running the whole oscillator sweep at
     128 substeps -- roughly 15 h, and it would multiply every published SLB cost there by
     four, turning the $302\times$ advantage at dim 128 into $\sim75\times$. Paying that
     to gain one dot is a bad trade. **It establishes reach, not cost scaling**, and is
     quoted here rather than plotted for that reason.
   - **What stops the study at dim 128 is certification, not propagation.** A dim-256
     reference at 128 substeps does run -- it completed in ~2.4 days (job 19559986). But
     certifying it requires agreement with a second run at a different resolution, and the
     cheap downward comparison at 64 substeps is itself unstable there. Checking upward
     instead costs ~2x the primary, putting a *certified* dim-256 oscillator reference at
     roughly a week. The propagation is affordable; the proof that it has converged is not.
   - `certified_reference` now escalates the check upward rather than discarding a good
     reference when the halved comparison diverges.

---


### 5.3 Provenance of Results 1-4

**All four Results now run on data regenerated under 0.6.4**, so every operator
count in this section matches the shipped code. Result 4 was the last to be
regenerated (Slurm jobs 19587109, 19587803 and 19587111, 2026-08-12/13); its
metadata records `degeneracy_tol = 1e-10`, the shipped default, on all three
systems.

**Why the regeneration mattered, stated plainly because it cuts against the
method.** The 0.6.4 floor removed operators contributing $10^{-24}$ relative or
less, so the dissipator is unchanged to double precision and every *accuracy*
conclusion from the older data survived. The *cost* comparisons did not:

> **Removing the spurious operators made `mcsolve` faster.** Its cost scales
> with `N_L`, since every jump evaluates all `N_L` jump probabilities. On the
> chain at dim 64 that is 0.174 s per trajectory at the old `N_L=113` against
> 0.056 s at the corrected 31 — a 3.1x speed-up, tracking the 3.6x drop in
> operator count. Bundling also lost headroom, since `M` can never exceed
> `N_L`: the old data allowed `M` up to 113, the corrected construction caps it
> at 31.

The pre-0.6.4 inputs are kept under `data/legacy/` rather than deleted, so the
older figures remain reproducible and the difference is auditable.

**Wall-clock comparability.** Times are comparable *within* a figure, where all
methods ran in one Slurm allocation on one node, and **not between** figures.
Result 4's three systems ran on landau29, landau92 and landau51; its speedup
*ratios* are sound, its absolute seconds are not directly comparable across
panels. Every data file records its hostname, job ID and thread settings for
exactly this check — `plot_method_comparison.py` refuses to draw a cost axis
across files that disagree unless forced.

### Result 1 — convergence dynamics versus the bundle size $M$

To see exactly *how* SLB converges to the exact solution as the bundle size $M$ grows, we can plot the time-evolution of several observables for each system. The dashed black line is the exact reference dynamics, and the colored lines are SLB at $M \in \{1, 2, 4, 8, 16, 32\}$, darkening as $M$ increases.

![System A convergence](convergence_dynamics_spin_chain.png)
![System B convergence](convergence_dynamics_mixed_chain.png)
![System C convergence](convergence_dynamics_oscillator_bath.png)

These plot $\langle O(t)\rangle$ against the exact reference as the system relaxes. As $M$ grows, the bundled mean tightens onto the reference—the approximation is a dial, not a fixed compromise. 

The systems differ dramatically in how fast they converge. The oscillator (System C) and mixed chain (System B) sit essentially on the reference at $M=8$, while the TFIM chain (System A) still shows visible deviation even at $M=16$. Convergence speed is set by the spread of the individual operator contributions and cross-terms, not by dimension alone, so it is worth checking on your own system.

**Beyond energy: capturing coherence.** Energy is nearly diagonal in the energy eigenbasis, so matching $\langle H\rangle$ says little about off-diagonal structure. Notice the `coherence` panels: SLB tracks the magnitude of the off-diagonal density matrix elements with the same convergence in $M$. This confirms SLB reproduces the full density matrix, not merely its diagonal.

**Sizes.** Every result in this section is available at Hilbert dimensions 16,
32 and 64 on all three systems, computed once per size and stored separately
(`accuracy_vs_M_<system>_dim<D>.json`); the plot script's `PLOT_DIM` selects
which to draw. Past dim 32 `mesolve` can no longer build its superoperator
here, so the reference at dim 64 is the certified native full-dissipator route
(§2), and the oscillator's dim-64 point runs at 16 RK4 substeps — disclosed,
because its stiffness demands it. The convergence laws survive the jump: on
the chain at dim 64 ($N_L=31$) the energy
bias still falls as $M^{-0.98}$ and the
statistical spread as $M^{-0.73}$, essentially unchanged from dim 16. On the
oscillator at dim 64 the bias is *comparable to* the sampling floor rather than
cleanly above it — it sits below at $M=4$, 32 and 64 and above at $M=2$, 8 and
16 — so the fitted slope there is not trustworthy and the individual points
should be read as upper bounds. At dim 32 the bias stays measurable at every
$M$, and the fit is meaningful.

**Why the oscillator's traces look featureless.** On the oscillator the SLB
curves sit on top of the reference at every $M$, with bands too narrow to see —
the figure appears to show nothing. That *is* the result: even $M=2$ tracks a
trajectory spanning $\langle H\rangle\approx13\to4$ to within $\sim\!10^{-2}$,
so there is no visible discrepancy to plot. It is the same fact that Results 2
and 4 report quantitatively (a handful of bundles suffices at every size —
$M^\ast\le 4$ under Result 2's tightened target — and the speedup over
`mcsolve` reaches three to four orders of magnitude): this system's dense,
stiff dissipator is exactly the structure bundling exploits. The error
decomposition below is where the oscillator's behaviour becomes legible.

**The anatomy of the error at its worst moment.** The time traces above show
*that* the estimate converges; this figure shows *how*. For each observable,
$t^\ast$ is the instant where the smallest $M$ estimate's $\mathrm{RMSE}(t)$ peaks — the
hardest moment of the dynamics — and is then held fixed for every $M$. At that
one instant the error splits into its two parts: the **bias**
$|\text{mean}(t^\ast)-\text{ref}(t^\ast)|$ and the **fluctuation** (the std over
realizations). The realization count is the same at every $M$ — nothing about
the sampling is tuned — so the trends are purely the effect of $M$: the bias
should fall like $1/M$ (the bundling systematic) and the fluctuation like
$1/\sqrt{M}$ (the bundling noise). On the chain the energy shows exactly this
($M^{-0.98}$ and $M^{-0.73}$ fitted at dim 64), and the pattern holds across
every system and size measured: the bias exponent lands between $-0.92$ and
$-1.09$, the fluctuation exponent between $-0.46$ and $-0.73$. One honest caveat: once the true bias drops
below the statistical floor of the run-mean (SEM $=$ fluctuation $/\sqrt{200}$),
the *measured* bias flattens into that noise — visible for the coherence at
large $M$, where the fitted bias slope is shallower for exactly this reason.

![System A error decomposition](benchmark_error_decomposition_spin_chain.png)
![System B error decomposition](benchmark_error_decomposition_mixed_chain.png)
![System C error decomposition](benchmark_error_decomposition_oscillator_bath.png)

**Construction is not dynamics.** The figure captions now report the two costs
separately: building the $N_L$ Davies/Lindblad operators (an eigendecomposition
plus $N_L$ operator assemblies — milliseconds at this size) versus propagating
the dynamics (seconds). The distinction matters at scale: construction grows
with its own exponent as the dimension increases, so the pipeline price should
never be blurred into the solve time — Result 2 now tracks it as its own cost
curve.

**Size invariance — of the slope, not of the height.** Overlaying the bias
sweep at every measured dimension on one axis separates two questions that are
easy to conflate.

**The slope is invariant, and that is the useful part.** Every curve falls as
$M^{-1}$, and the exponent settles as dimension grows. System A is measured at
**four** sizes: $-0.91$, $-0.97$, $-0.98$, $-0.98$ at dims 16, 32, 64, 128. It
stops moving after dim 32 and stays put through a further fourfold increase.
System B gives $M^{-1.00}$, $M^{-1.00}$, $M^{-1.02}$ across its three sizes. So
**doubling M halves the error at any size**, and that rule needs no
recalibration as the system grows.

**The height is not invariant, and the difference is systematic.** At fixed
$M=8$ the bias moves with dimension:

| system | dim 16 → 32 → 64 | scaling |
|---|---|---|
| **A** TFIM chain | $3.3\times10^{-2} \to 5.0\times10^{-2} \to 8.0\times10^{-2} \to 1.2\times10^{-1}$ | $\sim N^{+0.64}$ |
| **B** mixed chain | $2.6\times10^{-2} \to 3.4\times10^{-2} \to 5.5\times10^{-2}$ | $\sim N^{+0.55}$ |
| **C** oscillator | $2.6\times10^{-3} \to 2.3\times10^{-3} \to 2.0\times10^{-3}$ | $\sim N^{-0.20}$ |

System A's exponent is measured over four dimensions and the fourth does not
move it: $N^{+0.64}$ over dims 16–64, and still $N^{+0.64}$ with dim 128
included. The growth is a trend, not a three-point artefact.

On the chains the bias grows as roughly $\sqrt N$, so holding a fixed *accuracy*
target requires $M$ to grow with the system — which is precisely what Result 4's
iso-accuracy curve measures, and why its curve sits above the fixed $M$ one. The
oscillator is the genuinely invariant case: flat, in fact slightly improving
with size. **A fixed bundle count buys a fixed *convergence rate* at any
dimension; it buys a fixed *error* only where the structure is favourable.**

**The fluctuation.** SEM curves fall as expected on the chains — time-averaged
exponents of $-0.52$ to $-0.58$ against the predicted $M^{-1/2}$. The steeper
values in the legends ($-0.58$ to $-0.73$) come from evaluating at the fixed
$t^\ast$ rather than across the trajectory, and are an artefact of that choice.
The oscillator is the exception: it falls at $M^{-0.77}$ to $M^{-1.05}$ either
way, genuinely faster than $1/\sqrt M$. A plausible cause is its cross-term
structure — only 11% of its operator pairs produce a surviving cross term
(§2.6), so the bundling noise is not a sum of many independent contributions and
central-limit scaling need not hold. **That explanation is untested.**

Fitted exponents are quoted only where they clear the strict noise floor; with
the full 200 realizations every point does, on all three systems at all three
sizes.

![System A size invariance](accuracy_vs_M_invariance_spin_chain.png)
![System B size invariance](accuracy_vs_M_invariance_mixed_chain.png)
![System C size invariance](accuracy_vs_M_invariance_oscillator_bath.png)

### Result 2 — cost scaling versus the exact solver

![System A cost scaling](benchmark_cost_scaling_spin_chain.png)
![System B cost scaling](benchmark_cost_scaling_mixed_chain.png)
![System C cost scaling](benchmark_cost_scaling_oscillator_bath.png)

**Reading the lower panel.** It answers one question: at the bundle size chosen
above, is the remaining error *systematic* or *random*? The bar splits the mean
squared error into **bias²** (solid, dark) and **Std²** (light). The distinction
is practical rather than decorative — averaging more realizations shrinks the
light part and does nothing at all to the dark part, which only a larger $M$
removes. So a light bar means "just sample more", and a dark bar means "this is
as good as this $M$ gets".

Every swept dimension appears. Hatched bars are dimensions where the accuracy
target was **not** reached; they are drawn at the largest $M$ the sweep tried,
and they are the informative ones, because the split explains *why* the target
was out of reach. On System A the bias share climbs from 17% at dim 4 to 55%
by dim 64 and stays there — the error stops being noise you can average away
and becomes a systematic floor set by $M$, which is capped at $N_L$. That is the
same "nothing to compress" conclusion the rest of this page reaches, arrived at
from the error budget rather than from the operator count. On System C the bias
share instead *grows with a falling* $M^\ast$ — 0% at dim 16, 26% at 32, 51% at
64 as $M^\ast$ drops 8 → 8 → 4 — which is the method trading a little bias for a
lot of cost, deliberately, and still meeting the target.

The figure has two panels sharing the dimension axis. **Top:** wall-clock time
for one solve versus Hilbert-space dimension $N$. **Bottom:** the accuracy of the
SLB solve at each size, so the speed claim is qualified by the error it holds.
The dashed vertical line marks where one full `mesolve` exceeds the time budget —
past it the exact solver is impractical.

**The cost curves (top).** The exact full-dissipator `mesolve` evolves the
density matrix with all $N_L$ collapse operators; its fitted slope is the
steepest on the plot ($N^{4.9}$ on the chain, $N^{5.0}$ on the oscillator), and
past dim 32 it cannot run here at all — its superoperator construction exhausts
even 32 GB. SLB at a *fixed* bundle size ($M=8$) only ever propagates $M$
operators, so one solve is cheap; the SLB and construction curves need no exact
reference, so they extend well past the wall — on the oscillator all the way to
dim 128, a $16\times$ span in dimension.

One caveat on the SLB slopes, stated plainly because a fitted exponent invites
it: **the fixed $M$ and iso curves are clean power laws on the chain but not on
the oscillator.** On the chain the fixed $M$ cost fits $N^{2.5}$ over dim
4–512 with monotone per-step ratios, close to the $O(N^3)$-per-solve floor once
overhead is amortized — a quotable scaling law. (This number has moved twice and both moves were upward. It fitted
$N^{1.6}$ over dim 4–256 before the sweep reached 512, and $N^{1.9}$
before the fitting range was corrected. Both earlier values were
flattered by including the smallest dimensions, where the measured time
is interpreter and allocator overhead rather than the algorithm — on
this system `mesolve` costs 0.032 s at dim 4 and 0.021 s at dim 8, *less*
at the larger size, which cannot be a cost that grows with $N$. Every
slope on these panels is now fitted only over the curve's monotone tail
above a 0.1 s floor, and each legend entry states how many points that
left. Where only two survive, the number is labelled a local slope
rather than an exponent, because a line through two points is one ratio,
not a scaling law.) On the oscillator the same
curve fits $N^{1.7}$, but the per-doubling cost ratios are *not* monotone (a
large jump at dim 8→32, then a much smaller one at 32→64 as the dense linear
algebra reaches its efficient BLAS regime), so that number is a least-squares
summary of a curved trend, not a scaling exponent, and is **not quoted as
one**. The scaling *claim* of this work therefore rests on the chain's
$N^{2.5}$; the oscillator panel is included as the decisive visual of the exact
solver's wall — `mesolve` at $N^{5.0}$ and the native route at $N^{3.1}$ both
climbing into hours per solve while SLB stays near-flat — rather than for a
fitted SLB exponent.

**Where the oscillator's curves stop, and why it is stiffness rather than
memory.** All three end together at dimension 128, and the limit is the
integrator, not the operator count.

The panel is a *uniform-substep* benchmark: **every dimension** integrates at
one resolution, or the wall-clocks are not slope-comparable. Uniform across
dimensions, not across methods — the native curve is the reference-grade run and
carries a deliberate $2\times$ substep margin over SLB, so the SLB-vs-native
ratio should be read as against a *certified* exact solve rather than against
the same integrator. That margin is what makes the reference trustworthy, and it
inflates the ratio by roughly $2\times$; see §3.4 for the full accounting. Running it at 16 substeps,
the bundled generator diverged at dimension 128 — entries growing to $10^6$
while still finite. Re-running the whole panel at 32 substeps recovers that
dimension and pushes the failure to 256, where it diverges again, this time to
$4\times10^{17}$.

**That is a pattern, not a setting.** Each doubling of dimension needs roughly
double the substeps, because the anharmonic ladder's level spacing grows with
the level index: adding Fock states adds *faster* frequencies, so the generator
becomes stiffer as the system grows. Two consequences follow. The oscillator's
true cost per solve grows faster than the $M N^3$ the arithmetic suggests, since
the integrator must also take more steps. And a fixed-resolution comparison has
a natural ceiling — not because bundling fails at dimension 256, but because
measuring it there on the same axis as dimension 8 stops being meaningful.

The exact reference is subject to the same physics and shows it: at dimension
128 it needed 32 substeps, and its certification escalated **upward** to 64 when
the coarse partner diverged, agreeing to $3.6\times10^{-8}$. At dimension 256 it
too runs out — 2,986 operators against a reference that would need finer
integration still.

The gap this leaves is honest rather than missing: **bundling at dimension 256
is possible, it is simply not measurable on this axis** without changing the
integrator, which would change what the axis means.

**A second exact route, as a control.** The dash-dot curve is the same Lindblad
equation propagated by the package's own fixed-step RK4 with *all* $N_L$
operators — no bundling, no stochastic sampling, and no superoperators, so
memory stays proportional to the operator list rather than exploding. It is here
for one methodological purpose: **to supply the accuracy reference past the
point where `mesolve` can no longer provide one.** Wherever both routes run they
agree to $10^{-10}$ to $10^{-8}$ (stated in the figure footer, recorded per
dimension in the data), and where the native route is used alone it is re-run at
half its substeps and rejected if halving moves the answer appreciably. That it
also scales better than `mesolve` ($N^{2.5}$ / $N^{4.9}$) is worth noting but is
not a claim of this work: `mesolve` remains the neutral, widely-used standard
against which the cost argument is made. The
oscillator is the stiffer system: its anharmonic ladder's frequencies grow
like $n^2$, so the fixed-step RK4 needs more substeps to stay stable than the
chain does. Because a fixed-step cost curve is only slope-comparable at
*uniform* substeps, the whole oscillator sweep is run at a single elevated
count (32 substeps, recorded in the metadata and the footer) chosen for
stability at the largest size reached; at the smaller sizes this over-resolves
the dynamics, so those points are a mild upper bound on SLB's true cost — a
conservative bias, never a flattering one. At that setting every curve reaches
dim 128, a $16\times$ dimensional span, where the sweep's configured sizes end.
The committed dimension-128 reference points were generated with
`--native-ref-max 128`; `meta.params.native_ref_max_dim` records that override
alongside the reference substeps. The default remains 64 because extending the
reference is an hours-scale opt-in action, whereas the larger timing-only SLB
points are inexpensive.
The ladder's stiffness grows like $n^2$, so each further octave would demand
roughly $4\times$ more substeps; the principled route to larger sizes is an
implicit solver, not more RK4 steps (see the outlook). This same "stiffer than the physical system"
effect is why bundling itself needs those substeps: concentrating the
dissipative weight of all $N_L$ operators into $M$ bundles makes the realized
SLB generator **stiffer than the physical one**, increasingly so at small $M$.
Just below the hard stability cliff this produces samples that grow
exponentially while remaining finite — numerically meaningless yet invisible to
a plain finiteness check — which the solver's magnitude-and-trace guard now
catches against the physical scale and records (`slb_unstable_at_substeps` in
the data). The *reference* is held to the same standard: where it is produced
by the native route it is re-integrated at a different substep count and
rejected if the answer moves appreciably (tolerance $10^{-4}$). The cheap check
halves the substeps, but near the stability edge the halved run itself
diverges — which says nothing about the reference — so the benchmark then falls
back to the standard *upward* convergence test against a **finer** run at
double the substeps, recording which direction was used. At dim 128 this upward
fallback certifies the 32-substep reference against a 64-substep run to
$4\times10^{-8}$. Two costs that
must not be blurred are shown separately: the dotted curve is the one-time
**Davies construction** of the $N_L$ operators (an eigendecomposition plus $N_L$
operator assemblies) — cheap in absolute terms here, but scaling with its own
exponent. Inside each SLB realization there is also a bundle-assembly step
(combining all $N_L$ operators into $M$ bundles, cost $\sim M N_L N^2$, i.e.
$\sim N^4$ once $N_L \sim N^2$): an implementation term, not part of the
method's $O(N^3)$ propagation, and the natural target for a vectorized or
sparse bundle build if the top-end slope needs flattening. The iso-accuracy
curve is the honest one — read on.

**Fixed $M$ is cheap, but its accuracy decays with size.** At a
fixed bundle count the RMSE against the exact solve *grows* with $N$: $N_L$ grows
with the system, so a fixed number of bundles resolves the dissipator less
finely (the recorded sweep shows the fixed $M$ RMSE climbing through the
target as $N$ grows — the $M^\ast$ annotations on the iso-accuracy curve are
that mechanism made visible). The MSE-budget bars under the cost panel show
*why* the iso curve has the slope it does: at every $M^\ast$ the error is
bias-dominated (statistical noise is a single-digit share by dim 16), so the
slope is the bundling physics — $M^\ast$ must grow to cut bias — and more
sampling cannot flatten it. A pure fixed $M$ speed plot compares at a *moving*
accuracy, which invites the obvious objection: fast is meaningless if the error
blows up with $N$.

**Iso-accuracy — the cost to hold a *fixed* accuracy (third curve).** To answer
"fast *at what accuracy*", the iso-accuracy curve chooses, at each $N$, the
smallest bundle size $M^\ast$ — the first on the geometric grid
$M = 1, 2, 4, \ldots$ whose 16-run time-averaged RMSE reaches a fixed target (here
$\text{RMSE}=0.02$, measured against the exact solve) — and plots the cost of *that*
solve; each $M^\ast$ label on the iso-accuracy curve names the bundle size
paying that point's cost, and its achieved RMSE hugs the target from below in
discrete steps because $M$ is searched on a grid. (The run script records the
whole sweep, so the target defining $M^\ast$ is applied at analysis time — it
can be changed and the figure redrawn without re-running the benchmark.)
Past the wall the iso-accuracy curve keeps going: the accuracy reference there
is obtained by propagating the *full* dissipator with the package's native
fixed-step RK4 at doubled substeps (no superoperators, so no memory blow-up),
cross-validated against `mesolve` at the last size where both exist (agreement
at the $10^{-10}$ level; recorded in the data). Only the *reference* uses this
route — the red cost curve remains genuine `qutip.mesolve`. The dashed red
guide continues the exact solver past its feasibility wall at its own fitted
slope: measured against it, SLB's fixed $M$ point at the largest
dimension shown is cheaper by four to five orders of magnitude (chain, dim
256: an extrapolated $\sim\!3$ weeks versus a measured minute) — this, not
the sub-wall region where both methods are cheap, is the figure's claim. The required
$M^\ast$ grows with $N$ but *sublinearly*: measured on the chain across six
dimensions (4 to 128, the last two reached via the native reference), the
ladder runs $4\to16\to32\to32\to64\to64$ — close to $M^\ast\sim\sqrt{N}$, and
far short of the $M^\ast\propto N$ that would cost SLB a full power of $N$.
This is what keeps the chain's iso-accuracy slope near $N^{2.4}$ rather than $N^4$, so this curve is steeper than fixed $M$:
holding accuracy costs about one extra power of $N$. But it still sits far below
the exact solver, so SLB's advantage survives the honest accounting. It is
computable only up to the reference wall, since tuning $M^\ast$ needs the exact
**The target is chosen per system, and the three values are themselves a
result.** A target is not a claim about absolute accuracy; it fixes a common
operating point so that SLB and the exact route are compared doing the same job.
It has to be picked so the iso curve carries information, and what each system
requires differs by two orders of magnitude:

| | target | why that value |
|---|---|---|
| **A** TFIM chain | $0.05$ | **looser.** At $0.02$ this system misses at every dimension but the smallest — $M$ can never exceed $N_L$, and even $M=N_L$ parks between $0.024$ and $0.029$ across dim 8–512. |
| **B** mixed chain | $0.02$ | discriminating as it stands: $M^\ast$ climbs $4\to64$. |
| **C** oscillator | $0.005$ | **tighter.** $M^\ast=1$ clears $0.02$ at every size, so the looser target measured nothing. |

**Read System A's row as the finding it is.** It needs a target three times
looser than System B and ten times looser than System C — and even then
$M^\ast$ tracks $N_L$ almost exactly: $1/3$, $4/7$, $8/13$, $16/21$, $31/31$,
$32/43$, $57/57$, $64/73$. At dims 64 and 256 the two are *equal*. Reaching a
loose target on this system costs essentially every operator, which is the
no-compression claim measured rather than asserted.

The slopes make the same point more sharply. System A's iso-accuracy curve fits
$N^{3.0}$ while its exact solver fits $N^{2.5}$: **holding accuracy fixed makes
bundling grow faster than solving exactly.** The two curves are converging, and
past some dimension the exact route simply wins. That is the clearest statement
available of where this method should not be used — and it is the control system
behaving exactly as a control should.

At $0.005$ the oscillator ladder is $8/4/4/2$ across dim 16–128 — nearly flat,
consistent with a system whose bundling bias barely grows with size, and the
reason its iso and fixed $M$ curves sit close together rather than the iso curve
rising above.

**Quote the target whenever quoting a speedup from these panels.** A cost at
$0.05$ is not comparable with a cost at $0.005$.

On the oscillator's iso curve the dim-8 point is drawn as a hollow marker: at
that smallest size the single-run RMSE reaches $0.005$ only at $M=8$, one step
past that dimension's swept range ($M\le 4$), so the target is not met within
the sweep and the point is flagged rather than silently placed. Every larger
dimension meets the target with a solid $M^\ast$.

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

### Result 3 — four-method accuracy versus cost

`run_method_comparison.py` executes **four solvers in a single Slurm
allocation** at each dimension, so every wall-clock is from the same node:

1. **Native RK4 (`native`):** Full-dissipator dense RK4 on the density matrix without superoperators. Serves as certified reference past `mesolve` limits.
2. **`mesolve`:** QuTiP's standard exact solver, constructing the full $N^2 \times N^2$ Liouvillian.
3. **`mcsolve`:** QuTiP's Monte-Carlo trajectory solver ($N_{\text{traj}} = 500$).
4. **SLB:** Stochastically bundled dissipators ($M=16, 32$, 16 realizations).

**Accuracy against cost.** Each method is a point in the (wall-clock, error)
plane, so "which method reaches this accuracy for the least compute" is read
off directly; lower-left is better. SLB traces a curve as $M$ grows, `mcsolve`
is a single fixed-budget point, and the exact solvers sit at their own cost
with error at the integrator floor. Shade darkens with dimension.

#### System C — oscillator (dim 64, $N_L = 890$)

![Accuracy versus cost, oscillator, energy](benchmark_comparison_oscillator_bath_energy.png)
![Accuracy versus cost, oscillator, x_sx](benchmark_comparison_oscillator_bath_x_sx.png)
![Accuracy versus cost, oscillator, coherence](benchmark_comparison_oscillator_bath_coherence.png)

| observable | SLB ($M=16$) error | `mcsolve` error | SLB/mc ratio | SLB speed vs native |
|---|---|---|---|---|
| `energy` | $5.56\times10^{-4}$ | $3.44\times10^{-1}$ | **620x better** | 3.4x |
| `n` | $2.70\times10^{-4}$ | $9.09\times10^{-2}$ | 337x better | 3.4x |
| `sz` | $4.65\times10^{-4}$ | $6.29\times10^{-3}$ | 13.5x better | 3.4x |
| `n2` | $7.90\times10^{-3}$ | $2.52\times10^{0}$ | 318x better | 3.4x |
| `x_sx` | $3.11\times10^{-3}$ | $4.45\times10^{-3}$ | **1.4x better** | 3.4x |
| `coherence` | $6.31\times10^{-6}$ | $1.22\times10^{-4}$ | 19.3x better | 3.4x |

One SLB realization at $M=16$ costs 2.3 s and reaches $6.0\times10^{-6}$
relative error on the energy, against 121 s for the exact full-dissipator solve
(**54x cheaper**) and 7,118 s for `mcsolve` at 500 trajectories (**3,100x
cheaper**, and **490x more accurate** on the energy). `mcsolve` is slow here
because every jump must evaluate all 890 jump probabilities.

**The 620x headline is real but observable-dependent.** On `energy`, `n`, and
`n2`, SLB's advantage over `mcsolve` is 300–620x — but those three observables
are effectively the same curve (shape correlation +0.989 to +0.999 against
$\langle H \rangle$). The independent quantities are `x_sx` and `sz`, where
SLB's advantage drops to **1.4x** and **13.5x** respectively. The `x_sx` figure
is included precisely because it is the harsher test.

#### System B — mixed-field chain (dim 64, $N_L = 2{,}017$)

![Accuracy versus cost, mixed chain, energy](benchmark_comparison_mixed_chain_energy.png)
![Accuracy versus cost, mixed chain, sx](benchmark_comparison_mixed_chain_sx.png)
![Accuracy versus cost, mixed chain, coherence](benchmark_comparison_mixed_chain_coherence.png)

| observable | SLB ($M=16$) error | `mcsolve` error | SLB/mc ratio | SLB speed vs native |
|---|---|---|---|---|
| `energy` | $2.52\times10^{-2}$ | $2.60\times10^{-2}$ | 0.97x (comparable) | 6.4x |
| `zz` | $1.52\times10^{-2}$ | $1.45\times10^{-2}$ | 1.05x (comparable) | 6.4x |
| `sx` | $2.07\times10^{-2}$ | $4.31\times10^{-3}$ | **4.8x worse** | 6.4x |
| `sz` | $2.75\times10^{-2}$ | $3.23\times10^{-2}$ | 0.85x (comparable) | 6.4x |
| `zz_per_bond` | $3.05\times10^{-3}$ | $2.90\times10^{-3}$ | 1.05x (comparable) | 6.4x |
| `coherence` | $1.35\times10^{-2}$ | $4.02\times10^{-4}$ | **33.7x worse** | 6.4x |

SLB runs **379x faster** than `mcsolve` and **6.4x faster** than the exact
solve. On `energy` and `sz` the errors are comparable. But on `sx` SLB is
4.8x worse, and on `coherence` it is **33.7x worse** — `mcsolve` resolves
off-diagonal density-matrix elements far better than the bundled estimator at
this $M$.

**At dim 128** ($N_L = 8{,}193$), `mcsolve` was not run: at dim 64 it already
took 4,372 s against 74 s for the exact solve (59x *slower* than solving
exactly), because every jump must test all $N_L$ collapse operators.
Extrapolating that to $N_L = 8{,}193$ exceeded the job budget. The dim-128
comparison is SLB against the exact solver only: one SLB realization at $M=16$
runs **547x faster** (4.46 s against 2,442 s) at ~92% accuracy.

#### System A — TFIM chain (dim 64, $N_L = 31$)

![Accuracy versus cost, TFIM chain, energy](benchmark_comparison_spin_chain_energy.png)
![Accuracy versus cost, TFIM chain, sx](benchmark_comparison_spin_chain_sx.png)
![Accuracy versus cost, TFIM chain, coherence](benchmark_comparison_spin_chain_coherence.png)

| observable | SLB ($M=16$) error | `mcsolve` error | SLB/mc ratio | SLB speed vs native |
|---|---|---|---|---|
| `energy` | $4.21\times10^{-2}$ | $7.47\times10^{-3}$ | **5.6x worse** | 0.1x |
| `zz` | $2.18\times10^{-2}$ | $7.56\times10^{-3}$ | 2.9x worse | 0.1x |
| `sx` | $5.30\times10^{-2}$ | $4.77\times10^{-3}$ | **11.1x worse** | 0.1x |
| `zz_per_bond` | $4.35\times10^{-3}$ | $1.51\times10^{-3}$ | 2.9x worse | 0.1x |
| `coherence` | $2.36\times10^{-2}$ | $1.04\times10^{-2}$ | 2.3x worse | 0.1x |

This is Control 1. Davies grouping collapses operators to 31, so the exact
solve costs the same as bundling ($M=16$). SLB provides no speed advantage
when $N_L$ is small, and **is worse than `mcsolve` on every observable** —
the bundling bias dominates when there are too few operators to compress.

#### The dynamics themselves

![Dynamics, TFIM chain](benchmark_comparison_dynamics_spin_chain.png)
![Dynamics, mixed chain](benchmark_comparison_dynamics_mixed_chain.png)
![Dynamics, oscillator](benchmark_comparison_dynamics_oscillator_bath.png)

### Result 4 — iso-accuracy cost versus dimension

Results 2 and 3 each leave half a question. Result 2 scales cost with dimension
but only against the *exact* solver; Result 3 races SLB against `mcsolve` but at
a fixed size. This figure asks the combined question: **at each dimension, what
does it cost each method to reach the same accuracy?**

At every $N$ both methods are given the same target — ensemble
$\mathrm{RMSE}=0.02$ against the exact solve — and asked for the cheapest
setting that reaches it. For SLB that is the smallest bundle count $M^\ast$; for
`mcsolve` it is the trajectory count $N_{\rm traj}^\ast=(S/{\rm target})^2$ from
its fitted per-trajectory spread. The vertical gap between the two curves is the
speedup. The lower panel checks the chosen SLB operating point by splitting its
MSE into systematic bias² and statistical SEM² against the target.

![System A iso-cost](benchmark_isocost_vs_dim_spin_chain.png)
![System B iso-cost](benchmark_isocost_vs_dim_mixed_chain.png)
![System C iso-cost](benchmark_isocost_vs_dim_oscillator_bath.png)

**The three systems give three different answers, and they are the answers §2.5
predicts.**

| | $N_L$ (dim 4→64) | $M^\ast$ (dim 4→64) | speedup at dim 64 |
|---|---|---|---|
| **A** TFIM chain | 3 → 73 (dims 4–512) | 2 → **73** | 3x |
| **B** mixed chain | 7 → 2,017 | 2 → **32** | **149x** |
| **C** oscillator | 32 → 890 | 1 → **1** | **12,955x** |

Read the middle column against the first. **$M^\ast$ is what the method costs;
$N_L$ is what it replaces**, and the ratio between them is the whole result.

**System A — the control, and it fails as designed, over four octaves.**
This system is now swept from dimension 4 to 512, and $M^\ast$ equals $N_L$
*exactly* at every size from 16 upward: 13 operators need $M^\ast=13$, then
21/21, 31/31, 43/43, 57/57, 73/73. Not approximately — equal, across a 32-fold
range in dimension. There is no compression because there is nothing to
compress, and that is now a measurement across four octaves rather than an
observation at two.

From dimension 32 up **the target is not reached at all**, even at
$M^\ast=N_L$ — visible in the lower panel, where those bars sit above the target
line. That is not a failure of the solver: at $M=N_L$ a bundle is still a random
mixture, not the exact operator set, so it retains sampling error the target is
tighter than.

**And the advantage does not grow.** The speedup over `mcsolve` is 2x, 5x, 5x,
4x, 3x across dims 16 to 512 — flat, within noise, over a 32-fold range. Compare
System B below, where the same quantity climbs from 1x to 149x over a *smaller*
range. A flat ratio is the signature of a system with nothing to exploit: both
methods pay for all $N_L$ operators, so both scale the same way and their ratio
is a constant. On this system the honest advice remains to use the exact
solver.

**System B — the advantage compounds with dimension.** $N_L$ grows 288-fold
across the sweep while $M^\ast$ grows 16-fold, and the speedup follows: 1x, 2x,
4x, 45x, **149x**. This is the generic case, and the trend is the point — the
gap widens because `mcsolve` pays for every one of the 2,017 operators at each
jump while SLB pays for 32. At dim 64 that is 60 s against 2.5 hours.

**System C — a single bundle, at every size.** $M^\ast=1$ from dim 8 to dim 64.
One bundle, drawn once, reaches an accuracy `mcsolve` cannot reach with 300,000
trajectories. The speedup runs 68x → 715x → 7,095x → **12,955x**, and at dim 64
that is 25 seconds against roughly four days. This is the ladder structure of
§2.6 paying off exactly where it was predicted to.

**Three caveats, all of which cut against the numbers above.**

*`mcsolve`'s trajectory counts are extrapolated past the measured range on
System C.* The fit needs $N_{\rm traj}^\ast$ of 23,000 to 301,000 where the
sweep measured a few hundred, so those points are marked capped and the
speedups are quoted as the cap allows. They are lower bounds in the sense that
the trajectories really would be needed, but the per-trajectory cost is
extrapolated, not measured, at that count.

*The three systems ran on different nodes* — landau29, landau92 and landau51,
jobs 19587109, 19587803 and 19587111. Wall-clock is comparable **within** each
panel, where both methods ran in one allocation, and **not across** panels. The
speedup ratios are safe; the absolute seconds are not directly comparable
between systems.

*System A's numbers are quoted at a target it does not meet.* Its $M^\ast$ at
dims 32 and 64 is the largest available, not a setting that reached 0.02, so its
2–4x is a comparison at unequal accuracy and should be read as an upper bound on
the advantage rather than a measurement of it.

### Result 5 — past the reference wall

Every result above compares SLB against an exact solve, which caps each study at
the dimension where an exact solve is still possible. That is the wrong place to
stop. The regime the method exists for is the one where the operator list does
not fit, and it had never been shown.

System B at dimension 256 has $N_L = 32{,}637$ Davies operators. Held as a list
of dense matrices that is **31.9 GB** — the operators alone, before a single
step of propagation. `mesolve_ensemble_davies` never forms it: each operator is
built, folded into the bundles, and discarded, so peak memory is a bounded chunk
buffer plus the ensemble's bundles and does not grow with $N_L$.

**What this can and cannot claim.** There is no exact solve at this size, so
there is no error to report, and "SLB ran at dimension 256" proves only that it
terminated. The run is therefore scored on three things that can be checked
*without* a reference, each of which the method could have failed.

![Extreme dimension](benchmark_extreme_dimension_mixed_chain.png)

**Check 1 — it converges in $M$, and in the predicted form.**

| $M$ | $\langle H \rangle$ at $t=5$ | change | ratio |
|---|---|---|---|
| 8 | −10.74198 | | |
| 16 | −10.78844 | 0.04646 | |
| 32 | −10.81238 | 0.02394 | **0.515** |

The claim is not that the numbers stop moving — a wrong answer can also stop
moving. It is that they stop moving *as* $1/M$, which is what the bias is
predicted to do, so each doubling of $M$ should halve the change. Measured
0.515 against a predicted 0.500. In the right panel of the figure this is the
straight line, and a curve there would have falsified it. The two independent
pairwise extrapolations to $M \to \infty$ give −10.8349 and −10.8363, agreeing
to 0.0014; the three-point fit intercepts at −10.8356.

**Check 2 — the integrator holds.** Max $|\mathrm{Tr}-1| = 4.4 \times 10^{-16}$
across all three sweeps. The bundled generator is Lindblad by construction, so
this is a test of the propagation at a size where nobody had run it.

**Check 3 — it relaxes to the thermal state.** This is the strong one, for two
reasons: detailed balance makes the Gibbs state stationary, and
$\mathrm{Tr}(H \rho_{\mathrm{Gibbs}})$ needs only the eigenvalues of $H$, which
the construction already computes. The reference is therefore *free at any
dimension the method can reach*, and it is independent of everything bundling
does.

```
started at   -10.2000
ended at     -10.8692
Gibbs        -10.8737     <- from eigenvalues alone
```

It covered **99.3%** of the distance, and it has genuinely stopped: over the
last 20 time units the curve is flat to $10^{-6}$, so the 0.0046 that remains is
a floor and not a curve still creeping toward the line.

**What that residual does and does not settle.** The thermal run used 4
realizations rather than the sweep's 16, which puts its own standard error at
about 0.0040 — so the gap is **1.1 s.e.m.**, drawn to scale in the figure's
inset. At that size the data cannot distinguish landing exactly on the Gibbs
state from landing a little short of it, and this write-up does not claim the
former. What it does exclude is a gross failure: a bundled generator that had
lost detailed balance would not arrive within 0.7% of a number it was never
given.

One measured detail worth recording without explaining it: the long-time gap
(0.0046) is five times *smaller* than the $M=32$ bias in the transient window
(0.023 against the extrapolated −10.8356). Whatever bias survives into the
stationary state is smaller than the bias on the way there. That is an
observation from one system at one size, not a result.

**Why System B and not System A.** System A has a $Z_2$ symmetry, hence extra
conserved quantities, so it relaxes to a *symmetry-restricted* stationary state
rather than the global Gibbs state — measured at dimension 16 it settles at
−3.5054 against a Gibbs value of −3.4738. That gap is real physics, and it would
invalidate check 3 for reasons having nothing to do with bundling. System B has
no such symmetry.

**Cost.** 644.6 s, 1016.1 s and 1775.0 s for the three sweeps at 16 realizations
each, plus 4098.1 s for the thermal run — about 2.1 hours total on one node with
4 CPUs. Counting $N_L$ took 1.3 s and built no operators.

**Provenance.** Slurm job 19592847 on landau42, partition `roibq`, 2026-08-17.
Written by `run_extreme_dimension.py --size 8 --m-values 8 16 32`
(`benchmarks/slurm_extreme256.sh`), plotted by `plot_extreme_dimension.py`.

**Two caveats.** The thermal run's 4 realizations are thin, and a repeat at 16
would tighten the 1.1 s.e.m. statement considerably. And nothing here is an
*accuracy* measurement — Results 1–4 remain the only place error against a known
answer is reported, precisely because they stop where a known answer stops.

---

## 6. Validation and robustness

The checks that answer the obvious doubts.

> **Provenance, and why it does not undermine this section.** These validation
> figures were computed before the 0.6.4 Davies correction, from the
> `convergence_progress_*.json` files. Unlike §5, that does **not** put their
> conclusions in question: every claim here is about *convergence rates and
> accuracy* — how the bias falls with $M$, whether the jackknife steepens that
> rate, whether results move under a different seed or a finer integrator — and
> the 0.6.4 floor leaves the dissipator unchanged to double precision (it
> removes only operators contributing $10^{-24}$ relative or less, verified
> directly). Rates measured on an identical dissipator are identical.
>
> What *would* be affected is any cost statement, and this section makes none.
>
> Two real gaps remain: the mixed-field chain is not covered here at all, and
> the `mcsolve` fairness note below refers to Result 3.
> Both are omissions rather than errors.

**Convergence at the predicted rates, and the jackknife correction.** The
uncorrected bias should fall as $M^{-1}$ and the statistical spread as
$M^{-1/2}$ — both visible in every panel of the strips below (green and blue
curves), and the fitted bias exponent sits at $M^{-0.95}$ to $M^{-1.00}$
across all sizes, the strongest single check that the estimator behaves as
derived.

**Bias rate under the jackknife.** Sweeping $M$ at fixed size and comparing the
uncorrected estimator against the jackknife-2 one (same seeds at every $M$)
separates the correction's two effects: it lowers the bias *level* everywhere,
and — where the sampling floor is pushed low enough to see it — it **steepens
its rate** from $M^{-1}$ toward the $M^{-2}$ that the leading-order cancellation
the method predicts (Adhikari & Baer 2025; see `REFERENCES.md`). The three panels below, smallest to largest, show the
whole story at once: the jackknife rate is resolved and clearly steepened only
where enough points clear the noise floor.

![spin chain jackknife rate strip](benchmark_jackknife_rate_strip_spin_chain.png)
![oscillator jackknife rate strip](benchmark_jackknife_rate_strip_oscillator_bath.png)

| system | dim | uncorrected | jackknife | reduction | verdict |
|---|---|---|---|---|---|
| Spin Chain | 16 | $M^{-0.96}$ | $M^{-1.45}$ | 3.3–6.7× | rate steepens |
| Spin Chain | 32 | $M^{-0.96}$ | $M^{-1.78}$ | 3.0–9.7× | rate steepens |
| Spin Chain | 64 | $M^{-0.95}$ | $M^{-1.04}$ | 2.8–4.1× | level only |
| Oscillator Bath | 16 | $M^{-1.00}$ | $M^{-0.87}$ | 5.6–8.9× | level only |
| Oscillator Bath | 32 | $M^{-1.00}$ | — | 1.4–3.0× | marginal |
| Oscillator Bath | 64 | $M^{-0.97}$ | — | 2.1× | marginal |

Only where at least three points clear **twice** the SEM is a corrected rate
quoted; below that the corrected "bias" is Monte-Carlo noise and its ratio
inflates, so it is reported as an upper bound. Spin dim 32 (512 realizations)
is where the floor is low enough to see the full effect — the corrected bias
falls monotonically to $M^{-1.78}$, up to $9.7\times$ below the uncorrected one;
dim 16 shows the same steepening ($M^{-0.96}\to M^{-1.45}$). At the largest
sizes the realization budgets leave a higher floor, and the correction then
acts as a level reduction without a resolved change of law. The self-check in
`benchmark_convergence.py`, and the strip in `plot_jackknife_rate_strip.py`,
apply exactly this rule — steepening is claimed only where the data support it.
Reading across the panels of each strip, the uncorrected bias (green) also
rises with dimension while the jackknife keeps the corrected bias comparatively
flat — so Result 2's growth of the bias with system size is a known,
correctable effect, not a breakdown.

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

## 7. Reproducing and reading these numbers

The theory this builds on — the SLB method paper, the Lindblad/GKLS and Davies
foundations, and the QuTiP solvers benchmarked against — is collected in
[`REFERENCES.md`](../REFERENCES.md).

Absolute times depend on the machine, core count, and BLAS build — treat them as
relative comparisons. A few notes:

- `mcsolve` parallelizes trajectories across cores; both Result 3 and the
  four-method comparison (Result 3) pin it single-threaded to match SLB's serial
  loop. State the core count when reporting.
- **Wall-clock is only comparable within one job on one node.** The same
  certified dimension-256 reference took 2,744 s standalone and 86.6 s inside
  a sequential sweep on the same machine, and repeat measurements on a shared
  node vary by tens of percent — so ratios below ~1.5x are not resolved
  without repeats. Every run records its hostname and Slurm job id in the
  metadata; `plot_method_comparison.py` refuses to draw a cost axis across
  files that disagree.
- The first solve of any method pays one-time import/compile costs; discard a
  warm-up run for careful timing.
- Within each figure the system size, time grid, tolerances, and reference are
  identical across methods, so each plot is internally apples-to-apples.
- The full `mesolve` reference is "exact" only for the Davies–Lindblad model
  defined here; the Davies/secular construction is itself a weak-coupling
  approximation to the underlying open-system dynamics.
- Every figure is drawn from a `data/*.json` file written by the matching
  `run_*.py` script (metadata inside: package versions, seeds, parameter
  grids, timestamps), so any figure is traceable to the exact run that
  produced it — and re-styling or re-targeting a figure never requires
  re-running the benchmark. [`data/README.md`](data/README.md) lists the
  canonical filenames and separates them from superseded data.
  `python export_csv.py` flattens every canonical data file into
  Excel-friendly CSVs under `data/csv/`: the observable dynamics over time
  with their std in tidy long format, plus the scalar summaries.
