# References

The theory, methods, and comparison solvers this package builds on. Citation
details were checked against the publishers' records.

## The method implemented here

- S. Adhikari and R. Baer, *Stochastically Bundled Dissipators for the Quantum
  Master Equation*, Journal of Chemical Theory and Computation **21**,
  4142–4150 (2025). https://doi.org/10.1021/acs.jctc.5c00145

  The stochastic Lindblad bundling (SLB) scheme this package implements: the
  dissipator is quadratic in the collapse operators and so is reproduced in
  expectation by `M` random *bundles*, cutting the per-step cost from
  `O(N^5)` to `O(N^3)` with `M` independent of the Hilbert-space dimension.

## The Lindblad / GKLS master equation

- G. Lindblad, *On the Generators of Quantum Dynamical Semigroups*,
  Communications in Mathematical Physics **48**, 119–130 (1976).
  https://doi.org/10.1007/BF01608499

- V. Gorini, A. Kossakowski, and E. C. G. Sudarshan, *Completely Positive
  Dynamical Semigroups of N-Level Systems*, Journal of Mathematical Physics
  **17**, 821–825 (1976). https://doi.org/10.1063/1.522979

  Together these establish the general form of a Markovian, completely positive
  and trace-preserving generator — the equation SLB solves, and the form the
  bundled operators preserve.

## The Davies / secular construction of the collapse operators

- E. B. Davies, *Markovian Master Equations*, Communications in Mathematical
  Physics **39**, 91–110 (1974). https://doi.org/10.1007/BF01608389

  The weak-coupling (Davies) derivation that produces one collapse operator per
  Bohr frequency of the system-bath coupling. `davies_operators` builds exactly
  these; the operator count `N_L` growing with the number of transition pairs is
  the cost SLB is designed to tame. See `CONVENTIONS.md` for the sign and
  detailed-balance conventions used in this package.

## Standard reference text

- H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems*,
  Oxford University Press (2002).

  Textbook treatment of the Lindblad equation, the Davies/secular master
  equation, and the Monte-Carlo wave-function method — background for all of the
  above.

## The comparison solvers (QuTiP)

- K. Mølmer, Y. Castin, and J. Dalibard, *Monte Carlo Wave-Function Method in
  Quantum Optics*, Journal of the Optical Society of America B **10**, 524–538
  (1993). https://doi.org/10.1364/JOSAB.10.000524

  The quantum-trajectory unravelling that QuTiP's `mcsolve` implements — the
  Monte-Carlo solver SLB is benchmarked against in Result 3 and Result 4 of
  [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).

- J. R. Johansson, P. D. Nation, and F. Nori, *QuTiP: An Open-Source Python
  Framework for the Dynamics of Open Quantum Systems*, Computer Physics
  Communications **183**, 1760–1772 (2012).
  https://doi.org/10.1016/j.cpc.2012.02.021

- J. R. Johansson, P. D. Nation, and F. Nori, *QuTiP 2: A Python Framework for
  the Dynamics of Open Quantum Systems*, Computer Physics Communications
  **184**, 1234–1240 (2013). https://doi.org/10.1016/j.cpc.2012.11.019

  QuTiP itself, whose `mesolve` (exact Lindblad) and `mcsolve` (trajectories)
  are the two solvers this package extends and is measured against.
