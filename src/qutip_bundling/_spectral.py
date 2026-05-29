"""Internal helper: spectral data may be a callable or an aligned array."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

SpectralInput = Union[Callable[[float], float], np.ndarray, list, tuple]


def evaluate_spectral(
    spectral: SpectralInput,
    omegas: np.ndarray,
    name: str = "spectral function",
) -> np.ndarray:
    """Turn a callable or aligned array into values at every Bohr frequency.

    Parameters
    ----------
    spectral
        Either ``f(omega) -> float`` or an array-like of shape ``(N_L,)``
        whose i-th entry is the spectral value at ``omegas[i]``.
    omegas
        Bohr-frequency array of shape ``(N_L,)``.
    name
        Used only for error messages.

    Returns
    -------
    numpy.ndarray
        Real array of shape ``(N_L,)``.
    """
    omegas = np.asarray(omegas)
    if callable(spectral):
        values = np.array([float(spectral(float(w))) for w in omegas])
    else:
        values = np.asarray(spectral, dtype=float).ravel()
        if values.shape != omegas.shape:
            raise ValueError(
                f"{name} array has shape {values.shape}; expected {omegas.shape} "
                "(one entry per Bohr frequency)."
            )
    return values
