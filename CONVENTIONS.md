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
level ``a`` has ``omega > 0``. Paired with a detailed-balance spectral
function ``gamma(omega)`` that is **large at positive frequency**, this
makes energy-releasing (downward) transitions the fast ones, and the
system relaxes toward thermal equilibrium / the ground state at low
temperature.

If you flip this sign -- use ``omega = E_a - E_b`` while keeping the same
``gamma`` -- the dynamics run **backwards**: the system anti-relaxes and
heats up toward the *top* of the spectrum. This is a silent error; the
simulation still runs and conserves trace, it just evolves the wrong way.

**Recommendation:** build operators with :func:`davies_operators`, which
takes ``H`` and the coupling operator ``X`` and bakes in the correct
sign. Only use :func:`build_collapse_ops` (which trusts the ``omegas``
you supply) if you are constructing operators yourself and have checked
the sign.

A quick sanity check for any setup: start in an excited state at low
temperature and confirm the energy goes **down**.

## Detailed balance of gamma

For the equilibrium state to be the Gibbs state at temperature ``T``, the
spectral function must satisfy

    gamma(-omega) / gamma(+omega) = exp(-omega / kT)

The ohmic example used in the tests and demos satisfies this.

## Collapse-operator scaling

``build_collapse_ops`` returns ``c_alpha = sqrt(gamma(omega_alpha)) *
L_alpha``. That is the QuTiP convention: the collapse operator already
includes the rate, so the dissipator is

    D[rho] = sum_alpha ( c_alpha rho c_alpha^dag
                          - 0.5 { c_alpha^dag c_alpha, rho } )

with no extra ``gamma`` out front. If you have your own operators that do
*not* include the rate, multiply them by ``sqrt(gamma)`` before bundling.

## Lamb shift

The Lamb-shift Hamiltonian is built from the **bare** operators (without
the ``sqrt(gamma)`` factor and without bundling), as

    H_LS = sum_alpha imag_gamma(omega_alpha) * L_alpha^dag L_alpha

It is a coherent term added to the system Hamiltonian once; it is not
part of the dissipator and is never bundled. When ``imag_gamma`` is zero
it vanishes.

## Initial state

Nothing in the package forces an initial state, but note that reference
calculations often start from a *thermal* state of one subsystem rather
than a pure state -- e.g. an oscillator in its Gibbs state tensored with
a prepared spin state. Match whatever your reference uses; the bundling
method itself is agnostic to the initial condition.
