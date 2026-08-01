# CONTEXT.md — shared working context

**Shared by every AI tool working on this repository** (Claude Code, ChatGPT /
Codex, Antigravity, and any other). One file so no tool holds a partial picture.

## How to use this file

- **Read this file first**, before starting any task.
- **Append, never overwrite.** Add a new entry at the **bottom** of the session
  log. Do not rewrite or delete another tool's entry.
- **Sign and date every entry** with a heading in exactly this format:

  `## YYYY-MM-DD — <Tool Name> — <Short Title of Changes>`

  Add a time if more than one session happens in a day. Entries below that
  predate this convention keep their original headings.
- **Correcting someone else's entry:** do not edit it. Add your own entry
  saying what you found and why you disagree, and link the date you are
  correcting. The record should show the disagreement, not hide it.
- **Sections above the log** (project description, open questions, unresolved
  issues) are shared and *may* be edited by any tool — but say so in your log
  entry when you do.
- **Uncertain about something another tool did?** Put it under "Open questions"
  with your name, rather than guessing.

---

## What the project is

`qutip-bundling` implements the **stochastically bundled dissipator** of

> S. Adhikari and R. Baer, *Stochastically Bundled Dissipators for the Quantum
> Master Equation*, J. Chem. Theory Comput. **2025**, 21, 4142–4150.

A Lindblad master equation with `N_L` collapse operators costs one matrix
product per operator. The dissipator is *quadratic* in the operators, so it can
be reproduced **in expectation** by `M` random linear combinations
(`R_m = (1/√M) Σ_α r_{m,α} c_α`, with `E[r]=0`, `E[r r*]=1`). The bundled
dissipator is an unbiased estimator for any `M` and retains Lindblad form, so
the dynamics stay CPTP.

Version at time of writing: **0.6.4** plus unreleased changes.

> **Naming note.** "StochLind" is the **C++ reference implementation** this
> package is validated against — referenced in `CONVENTIONS.md` (the `sparcity`
> scalar) and `operators.py`, but its source is not in this repository. The
> package here is `qutip-bundling`.

### Architecture

```
src/qutip_bundling/
  operators.py       the method itself — pure operator transforms
                     davies_operators, build_collapse_ops, bundle,
                     bundle_from_phases, lamb_shift_hamiltonian,
                     prepare_bundled_dynamics, random_phases
  solvers.py         optional convenience — mesolve_ensemble,
                     mesolve_jackknife, BundledResult
  native_solver.py   rk4_mesolve: dense RK4 on the density matrix, avoids
                     building an N²×N² Liouvillian. SolverInstabilityError.
  _spectral.py       spectral-input handling (callable or array)
tests/               92 tests
benchmarks/          run_*.py (compute) / plot_*.py (figures) split;
                     common.py is the single source of truth for systems,
                     grids, metrics, observables, and metadata
```

Design principles worth preserving:

- **Operators in, operators out.** The core API returns operator lists, not
  solver results, so it composes with `mesolve`, `mcsolve`, `smesolve`, or a
  custom propagator.
- **Run/plot split.** `run_*.py` computes and writes JSON with full provenance;
  `plot_*.py` derives figures from that JSON in seconds. Accuracy targets are
  applied at *plot* time, so they can change without recomputing.
- **The Lamb shift is deterministic** and built from the bare operators; it is
  never bundled.

### Two benchmark systems

| | System A — spin chain | System B — oscillator |
|---|---|---|
| Model | transverse-field Ising, `J=1.0, h=0.6` | anharmonic oscillator + spin |
| Bath coupling | `X = Σᵢσˣᵢ` (collective) | `X = x⊗I` (oscillator only) |
| `N_L` | `n²−n+1` in sites — 43 at dim 128 | 408 at dim 32, 890 at dim 64 |

---

## The bundling / vectorization scheme

`bundle_from_phases` previously built each bundle with a Python loop of
`M × N_L` `Qobj` additions. It now stacks the operators once into an
`(N_L, N²)` array and forms every bundle with a single BLAS matrix product,
`(M, N_L) @ (N_L, N²)`. Output is identical to the explicit summation (max
deviation `8e-16`), pinned by 17 regression tests. Measured speedups on the
spin chain at `M=8`: **3.9× at dim 16, 2.5× at dim 32, 1.5× at dim 64** — the
win is loop overhead, so it shrinks as real arithmetic starts to dominate.

**Unresolved:**

1. **The vectorized path densifies everything** — an `(N_L, N²)` complex array,
   `N_L·N²·16` bytes. At dim 256 with `N_L=57` that is ~60 MB, fine. Under the
   pre-0.6.4 counts (`N_L=839`) it would have been ~880 MB. Nothing guards
   this; a model with genuinely large `N_L` at large `N` could exhaust memory
   where the old loop would not.
2. **Sparsity is discarded** — the result is always a dense `Qobj`. Usually
   correct for a sum of many operators, but unchecked.
3. **The deeper issue is `N_L` itself** — see below.

---

## The finding that reframes the method

*(established 2026-07-31, Claude Code — see log)*

A four-method comparison (native RK4 / `mesolve` / `mcsolve` / SLB) in one
Slurm allocation showed the bundling advantage is governed by **`N_L`, not
Hilbert dimension**:

- **Oscillator, dim 32, `N_L=408`:** SLB reaches `1e-3` error ~**27× cheaper**
  than the exact full-dissipator solve, and ~300× cheaper *and* ~100× more
  accurate than `mcsolve` at 500 trajectories. The method works.
- **Spin chain, dim 128, `N_L=43`:** SLB is **not competitive**. The exact
  solve costs about the same as the cheapest useful bundle and is seven orders
  of magnitude more accurate. Verified for single realizations as well as
  16-run ensembles, so it is not an artifact of ensemble cost.

Cause: bundling can only pay when `M ≪ N_L`. The chain's collective coupling
and ℤ₂ symmetry give `N_L = n²−n+1`, polylogarithmic in Hilbert dimension —
there is nothing to bundle.

This invalidated a prior README claim that bundling on the spin chain "is never
slower and typically a few-fold faster" than `mcsolve`. That claim rested on
pre-0.6.4 data where the chain's `N_L` was inflated to 113/325 by roundoff-only
projector blocks. The README now presents the chain as a control for where the
method does *not* help.

---

## Open questions

**From Claude Code, 2026-07-31:**

1. **Do the spin-chain Results 1–4 stay headline results?** Their data is still
   0.6.3 and carries the inflated `N_L`; it needs regenerating regardless, but
   whether those results are headline or control is an author decision and is
   currently blocking the regeneration run.
2. **Should this file be committed?** Currently untracked. The repository is
   public, so if it accumulates unpublished results it may belong in
   `.gitignore` instead. Not decided.
3. ~~**Where does `mesolve` actually stop being usable?**~~ **ANSWERED**
   2026-08-01 by job 19559694 — see the log entry for that date. Two separate
   walls at 32 GB: the chain fails at dim 128 on Liouvillian *size*, the
   oscillator at dim 64 on *operator count* (`N_L=890`).

---

## Session log

### Codex — 2026-08-01 03:21
- What I did: Read the shared context fully and reviewed the newer work already recorded here; made no project or scientific changes.
- Why: The repository context is substantially ahead of this conversation, so duplicating earlier work could conflict with established results and decisions.
- Files touched: `CONTEXT.md` only.
- New open questions (if any): None; the existing Open Questions remain unchanged.

### 2026-07-31 (into 2026-08-01) — Claude Code

**Library**
- Merged `bundle-vectorization` (`1b3b1ec`): BLAS rewrite of
  `bundle_from_phases`, +17 regression tests. Test count 75 → 92.

**Correctness fixes**
- `MAX_FULL_DIM` was a module constant used by four runners with no override —
  a cluster run silently recorded `t_full = NaN` above dim 32. Added
  `--max-full-dim` (`315e310`).
- `plot_high_dim_spin_reference.py` zipped curves against a four-colour
  palette, silently dropping every dimension past the fourth (`4960813`).
- `mcsolve` with `keep_runs_results` returns per-trajectory arrays, not an
  average; the comparison runner stored them unaveraged (`0e95481`).
- Comparison JSON written with default indentation; switched to
  `save_data(compact=True)`, added `compact_comparison_data.py` (`58e1f10`).

**New tooling**
- `run_method_comparison.py` — four methods on one footing (`e1c1ee9`).
- `plot_method_comparison.py` — accuracy-vs-cost and dynamics per observable;
  refuses to draw a cost axis across different Slurm jobs (`0e95481`).
- `build_high_dim_sheet.py` — regenerates the workbook's `High-dim Ref` sheet
  from JSON; that sheet had no generator and had drifted (`4960813`).
- `--repeats` for wall-clock error bars (`05b7293`).

**Data**
- Certified references d=4 → **1024**, `N_L = n²−n+1` confirmed throughout
  (`bc5dcdf`, `89301b1`).
- Four-method comparison, both systems, one allocation (`89301b1`).

**Docs**
- Missing 0.6.4 CHANGELOG entry; `N_L` reconciled across `BENCHMARKS.md`;
  `nohup` recipe replaced with `sbatch` (`b09e6ea`).
- README reframed around `N_L` rather than dimension (`ae611cb`).
- Root dev notes archived, checkout paths redacted (`061f323`, `78e1289`).

**Decisions and assumptions**
1. **The 0.6.4 block floor is numerically free** — verified, not assumed:
   against a reference keeping every nonzero block it differs by `1e-24`–`1e-33`
   relative in the dissipator superoperator, and *exactly zero* applied to a
   state. So accuracy-type results carry over; cost-type results do not,
   because `t_full` depends on the operator count.
2. **Did not rewrite the scientific narrative unilaterally** — fixed only
   unambiguous factual errors first (an `N_L=869` figure wrong under every
   version; a §2.3/§2.4 self-contradiction). Reframing came after approval.
3. **Added `--max-full-dim` rather than raising the global constant**, keeping
   the laptop-safe default and making the cluster opt-in explicit.
4. **Added an `execution` metadata block** (hostname, CPUs, thread env vars,
   Slurm job/nodelist/partition). This immediately caught a contaminated sweep.
5. **Observable design** — `⟨H⟩` alone is insufficient: it is built almost
   entirely from the diagonal of ρ, and on System B the spin carries under 4% of
   the total energy. Each system also gets the automatically chosen dominant
   coherence plus the individual terms of its own Hamiltonian, which reconstruct
   `⟨H⟩` exactly (residual `9e-16` chain, `2e-15` oscillator) and so double as a
   correctness check. The coherence operator is chosen once from the reference
   and shared by every method.
6. **`mcsolve` runs at a fixed `ntraj` budget**, reporting whatever accuracy
   that buys rather than being tuned to a target (author's decision).
7. **Cluster jobs ordered by value** — each dimension checkpoints to its own
   file, so a timeout loses only the tail, and the tail is always something
   already in hand.
8. **Repaired rather than re-ran** after the `mcsolve` averaging bug: the mean
   is exactly recoverable, so `compact_comparison_data.py` averages and rewrites
   (31.1 MB → 3.3 MB) instead of spending hours of cluster time.
9. **High-dimension figure shows all nine dimensions**, not a subset — the
   energy-density convergence toward a thermodynamic limit is itself a result.

**Process incidents worth remembering**
- Two sweep jobs overlapped on the same node (19558978 timed out while 19558993
  ran inside its window), contaminating all timings and leaving one dimension
  written by a different job running older code. Detected *only* because the
  execution metadata added this session records the Slurm job id. Physics was
  unaffected — trace distances were bit-identical across independent jobs — but
  the sweep had to be repeated. **Always check `squeue -u $USER` is empty
  before submitting.**
- Wall-clock varies ~30% between jobs even on the same node with the same
  thread count. Ratios below ~1.5× are unresolved without `--repeats`.
- The cluster shell is **tcsh**: bash `for ... do ... done` loops and multi-line
  quoted strings fail there. Use `git commit -m "subject" -m "body"` on one line.

**Left running:** job 19559694 (oscillator dim 64, timing repeats, `mesolve`
wall probe), submitted 2026-08-01 02:48.



---

## 2026-08-01 — Claude Code — Adopt shared logging protocol; note missing HANDOFF assets

**What I did**
- Adopted the entry-heading format specified by the author:
  `## YYYY-MM-DD — <Tool Name> — <Short Title of Changes>`, and updated the
  "How to use this file" section above to state it exactly. Earlier entries
  keep their original headings.
- Checked the workspace protocol's other two artifacts against the repository.

**What does not exist in this repository (as of 2026-08-01)**
- `HANDOFF.md` — absent. There is therefore no file of "canonical numbers" to
  keep consistent, and no defined relationship between it and this file.
- `scripts/check_record_consistency.py` — absent; there is no `scripts/`
  directory at all. The protocol says to run it before updating canonical
  numbers in `HANDOFF.md`; that step cannot be performed and I did not
  pretend otherwise.

  These may live in a different repository, or be planned rather than built. I
  did not create either, because inventing a consistency checker without
  knowing what it is supposed to check would be worse than not having one.

**Note on the entry above (Codex, 2026-08-01 03:21)** — recorded here rather
than by editing theirs, per the rule above. It uses `### Codex — <date>` rather
than the specified format, and was inserted at the **top** of the log rather
than appended at the bottom. Content is unaffected; noting it only so the
ordering convention does not quietly erode. Later readers should take the log
order as chronological *by heading date*, not by position, until the entries
settle into one convention.

**No code, data, or scientific changes this session.** Files touched:
`CONTEXT.md` only.

**Still open:** the three Open Questions above stand, and job 19559694
(oscillator dim 64, timing repeats, `mesolve` wall probe) was still running at
the end of the previous session.

---

## 2026-08-01 05:25 — Claude Code — mesolve memory wall measured; final comparison job submitted

**Result: `mesolve` has two independent memory walls, and they are different in
kind.** Measured at 32 GB on landau44 (job 19559694):

| system | dim | `N_L` | `mesolve` |
|---|---|---|---|
| chain | 32 | 21 | 5.71 s |
| chain | 64 | 31 | 221.26 s |
| chain | 128 | 43 | **OOM > 32 GB** |
| oscillator | 16 | 128 | 1.79 s |
| oscillator | 32 | 408 | 178.92 s |
| oscillator | 64 | 890 | **OOM > 32 GB** |

The chain fails on **dimension** — its Liouvillian at dim 128 is 16384², about
4.3 GB dense before temporaries. The oscillator fails at only dim 64, where the
Liouvillian is a mere 268 MB, because **890** of them must be summed. So the
oscillator hits a wall at a quarter of the chain's dimension, for an unrelated
reason. This is a sharper statement than "mesolve dies around dimension N" and
should replace any dimension-only phrasing in `BENCHMARKS.md`.

**Job 19559694 (OOM, partial).** My sizing error: `--max-full-dim 64` was
chosen from the chain, where dim 64 means `N_L=31`, but the oscillator at dim
64 has `N_L=890`. `mesolve` was attempted there and exhausted 32 GB. Because
the `sbatch --wrap` stages are separated by `;` rather than `&&`, the shell
continued past the kill, so stages 2 and 3 (timing repeats for chain dims
32/64 and oscillator dims 16/32) completed successfully. Worth remembering:
`;` between stages makes a multi-stage job partially recoverable.

**Job 19559720 submitted 05:25** — the definitive comparison run. Oscillator
dim 64 first (`--methods native mcsolve slb`, no `mesolve`, `--repeats 1`
because 3–4 h is too expensive to repeat), then everything else re-run with
`--repeats 3` so the whole dataset carries one job id. `mesolve` capped below
both walls. ~5 h expected.

**Note on provenance:** before this job, comparison files carried a mix of job
19559570 and 19559694. Per-dimension comparisons stayed valid (all methods
within a dimension shared a job) but cross-dimension cost trends did not.
19559720 is intended to make the whole set uniform. Check with the job-id
one-liner before trusting any cost axis; `plot_method_comparison.py` refuses to
draw one across mixed jobs unless `--allow-mixed-jobs` is passed.

**Shared section edited:** marked Open Question 3 (where `mesolve` stops being
usable) as answered, pointing at this entry. Flagged here per the rule that
edits above the log must be declared in the log.

**Files touched:** `CONTEXT.md` only. No code or committed data changed.
