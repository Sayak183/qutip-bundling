"""qutip-bundling: stochastic bundling of Lindblad operators for QuTiP.

Implements the stochastically bundled dissipator of Adhikari & Baer,
J. Chem. Theory Comput. 2025, 21, 4142-4150
(https://doi.org/10.1021/acs.jctc.5c00145).

Two layers:

* :mod:`qutip_bundling.operators` -- pure operator transforms
  (:func:`build_collapse_ops`, :func:`bundle`,
  :func:`lamb_shift_hamiltonian`, :func:`prepare_bundled_dynamics`).
  These are the method itself; they have no dependence on QuTiP's solver
  machinery beyond ``Qobj`` arithmetic. Compose them with ``mesolve``,
  ``mcsolve``, ``smesolve``, or your own propagator.

* :mod:`qutip_bundling.solvers` -- optional ``mesolve``-based ensemble
  and jackknife helpers, for users who just want averaged dynamics.
"""

from .native_solver import NativeResult, rk4_mesolve
from .operators import (
    BundledOps,
    build_collapse_ops,
    bundle,
    bundle_from_phases,
    davies_operators,
    lamb_shift_hamiltonian,
    prepare_bundled_dynamics,
    random_phases,
)
from .solvers import BundledResult, mesolve_ensemble, mesolve_jackknife

__version__ = "0.5.0"

__all__ = [
    # operator transforms (the method)
    "davies_operators",
    "build_collapse_ops",
    "bundle",
    "bundle_from_phases",
    "lamb_shift_hamiltonian",
    "prepare_bundled_dynamics",
    "random_phases",
    "BundledOps",
    # solvers
    "rk4_mesolve",
    "mesolve_ensemble",
    "mesolve_jackknife",
    "NativeResult",
    "BundledResult",
    "__version__",
]
