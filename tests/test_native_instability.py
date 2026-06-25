"""Tests for the native RK4 solver's instability guard.

The fixed-step RK4 propagator is only conditionally stable. When the Lindblad
generator is too stiff for the chosen step size, the density matrix overflows to
inf/NaN. ``rk4_mesolve`` must detect this and raise ``SolverInstabilityError``
rather than silently return a corrupted result.
"""

import numpy as np
import pytest
import qutip

from qutip_bundling import rk4_mesolve, SolverInstabilityError


def test_stable_solve_runs_and_is_finite():
    """A well-resolved solve returns finite expectation values."""
    H = qutip.sigmaz()
    rho0 = qutip.basis(2, 0)
    tlist = np.linspace(0.0, 5.0, 40)
    res = rk4_mesolve(H, rho0, tlist, [0.3 * qutip.sigmam()],
                      e_ops=[qutip.sigmaz()], substeps=4)
    assert np.all(np.isfinite(res.expect[0]))


def test_divergence_raises_solver_instability():
    """An extreme rate with a single coarse substep overflows RK4 to a
    non-finite state, which must raise SolverInstabilityError (not return NaN)."""
    H = qutip.sigmaz()
    rho0 = qutip.basis(2, 0)
    tlist = np.linspace(0.0, 5.0, 15)
    with pytest.raises(SolverInstabilityError):
        rk4_mesolve(H, rho0, tlist, [1e4 * qutip.sigmam()],
                    e_ops=[qutip.sigmaz()], substeps=1)


def test_instability_message_is_actionable():
    """The error names the remedy (increasing substeps)."""
    H = qutip.sigmaz()
    rho0 = qutip.basis(2, 0)
    tlist = np.linspace(0.0, 5.0, 15)
    with pytest.raises(SolverInstabilityError, match="substeps"):
        rk4_mesolve(H, rho0, tlist, [1e4 * qutip.sigmam()],
                    e_ops=[qutip.sigmaz()], substeps=1)
