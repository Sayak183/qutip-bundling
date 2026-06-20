# Conventions

A few conventions matter for getting correct physics out of this package.
The most important one bit us during validation against a reference C++
implementation, so it is spelled out first.

## Bohr-frequency sign (the one that matters)

A Davies/Bohr Lindblad operator connects two energy eigenstates,

    L = <a|X|b> |a><b|

and carries an associated Bohr frequency. The convention this package
uses (and that :func:`davies_operators` enforces) is

    omega = E_b - E_a

so that a transition from the **higher** level ``b`` to the **lower**
level ``a`` has $$\omega > 0$$. Paired with a detailed-balance spectral
function $$\gamma(\omega)$$ that is **large at positive frequency**, this
makes energy-releasing (downward) transitions the fast ones, and the
system relaxes toward thermal equilibrium / the ground state at low
temperature.

If you flip this sign -- use $$\omega = E_{a} - E_{b}$$ while keeping the same
$$\gamma$$ -- the dynamics run **backwards**: the system anti-relaxes and
heats up toward the *top* of the spectrum. This is a silent error; the
simulation still runs and conserves trace, it just evolves the wrong way.

**Recommendation:** build operators with :func:`davies_operators`, which
takes ``H`` and the coupling operator ``X`` and bakes in the correct
sign. Only use :func:`build_collapse_ops` (which trusts the $$\omega_{s}$$
you supply) if you are constructing operators yourself and have checked
the sign.

A quick sanity check for any setup: start in an excited state at low
temperature and confirm the energy goes **down**.

## Detailed balance of gamma

For the equilibrium state to be the Gibbs state at temperature ``T``, the
spectral function must satisfy

$$ \gamma(-\omega) / \gamma(+\omega) = exp(-\omega / kT)$$

The ohmic example used in the tests and demos satisfies this.

## Collapse-operator scaling

``build_collapse_ops`` returns $$c_{\alpha} = sqrt(\gamma(\omega_{\alpha})) *
L_{alpha}$$. That is the QuTiP convention: the collapse operator already
includes the rate, so the dissipator is

$$D[\rho] = \sum_{\alpha} ( c_{\alpha} \rho c_{\alpha}^{\dagger}
                          - 0.5 \{ c_{\alpha}^{\dagger} c_{\alpha}, \rho \} )$$

    

with no extra ``gamma`` out front. If you have your own operators that do
*not* include the rate, multiply them by ``sqrt(gamma)`` before bundling.

## Sparsity and thresholds

When the coupling operator ``X`` is sparse in the energy eigenbasis,
most Bohr pairs contribute a negligible operator and can be skipped.
This package exposes that as opt-in thresholds; every one defaults to
``0.0`` (keep everything), so leaving them alone reproduces the full,
unpruned operator set exactly.

Two distinct quantities can be thresholded -- they are *not*
interchangeable, which is why they are separate arguments:

* **Coupling element**, ``|<a|X|b>|``. Controlled by
  ``coupling_threshold`` in :func:`davies_operators`. A pair below this
  is dropped *before* ``gamma`` is evaluated or any matrix is built, so
  this is the knob that actually speeds up the build (the construction
  iterates only the significant entries instead of all ``N**2`` pairs).

* **Operator weight**, ``sqrt(gamma(omega)) * |<a|X|b>|`` -- equivalently
  ``sqrt(gamma) * ||L||``, the norm of the resulting collapse operator.
  Controlled by ``threshold`` in both :func:`davies_operators` and
  :func:`build_collapse_ops`. This is the master keep gate: it accounts
  for both how strongly two levels couple *and* how fast the bath drives
  that transition.

The reference C++ (StochLind) uses a single scalar ``sparcity`` applied
at several nested levels -- on the coupling element, on the rate, and on
their product. The two arguments here expose the two that matter in
practice; setting them equal mimics the C++ closely, but keeping them
separate lets you prune cheaply on coupling without committing to the
same cut on the weighted operator.

**Choosing a value.** Start at ``0.0`` and confirm your physics. Then
raise ``coupling_threshold`` until the operator count drops but the
observable you care about does not move beyond your tolerance -- the
dropped operators were contributing essentially nothing. A threshold a
few orders of magnitude below the typical ``|<a|X|b>|`` is usually safe;
too aggressive a value silently removes real dissipation channels and
changes the steady state. Always re-check against the unpruned result.

## Lamb shift

The Lamb-shift Hamiltonian is built from the **bare** operators (without
the ``sqrt(gamma)`` factor and without bundling), as

    H_LS = sum_alpha imag_gamma(omega_alpha) * L_alpha^dag L_alpha

It is a coherent term added to the system Hamiltonian once; it is not
part of the dissipator and is never bundled. When ``imag_gamma`` is zero
it vanishes.

The Lamb shift has its *own* threshold (``lamb_shift_threshold`` in
:func:`davies_operators`), because its terms are filtered against
``|imag_gamma(omega)|`` -- a different quantity from the operator weight
above. By default it inherits the operator ``threshold``, which means an
aggressive operator cut would also drop Lamb-shift terms by an unrelated
criterion. If you prune operators hard but still want the full Lamb
shift, set ``lamb_shift_threshold=0.0`` to decouple the two.

## Initial state

Nothing in the package forces an initial state, but note that reference
calculations often start from a *thermal* state of one subsystem rather
than a pure state -- e.g. an oscillator in its Gibbs state tensored with
a prepared spin state. Match whatever your reference uses; the bundling
method itself is agnostic to the initial condition.
