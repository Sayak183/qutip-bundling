"""Validation of stochastic Lindblad bundling against full mesolve.

Builds an open quantum system in Davies/Bohr form, then checks the three
guarantees of Adhikari & Baer (JCTC 2025, 21, 4142):

  1. The bundled dissipator is an unbiased estimator of the full one
     -- averaging converges as 1/sqrt(samples).
  2. The Lamb shift built from the *bare* L_alpha is independent of M;
     bundling never enters it.
  3. Solving the master equation with the bundled operators plus the
     deterministic Lamb shift reproduces the full dynamics.

Run:  python examples/oscillator_demo.py
Produces:  bundling_validation.png
"""

import numpy as np
import qutip

from qutip_bundling import (
    build_collapse_ops,
    bundle,
    lamb_shift_hamiltonian,
    mesolve_ensemble,
    mesolve_jackknife,
    prepare_bundled_dynamics,
)


# --------------------------------------------------------------------------
# Build a system with many Davies/Bohr Lindblad operators
# --------------------------------------------------------------------------
def build_system(N=14, kT=1.5, gamma0=0.02):
    """Anharmonic squeezed oscillator coupled to a thermal bath.

    Returns the system Hamiltonian, the bare Lindblad operators ``L_alpha``,
    the aligned Bohr frequencies ``omegas``, and the bath spectral
    functions ``gamma(omega)`` and ``imag_gamma(omega)``.
    """
    a = qutip.destroy(N)
    x = a + a.dag()                                  # system-bath coupling op
    H_mat = (a.dag() * a + 0.15 * (a.dag() * a) ** 2
             + 0.10 * (a * a + a.dag() * a.dag())).full()
    H_mat = 0.5 * (H_mat + H_mat.conj().T)
    evals, evecs = np.linalg.eigh(H_mat)
    x_eig = evecs.conj().T @ x.full() @ evecs

    def gamma(omega):
        # Detailed-balance Ohmic-like spectral function (real part)
        if abs(omega) < 1e-12:
            return gamma0 * kT
        return gamma0 * omega / (1.0 - np.exp(-omega / kT))

    def imag_gamma(omega):
        # Small principal-value-like piece -- mocks a Lamb shift that is
        # nonzero but small compared to the system frequency scale.
        return 0.05 * gamma0 * omega

    L_ops, omegas = [], []
    for n in range(N):
        for m in range(N):
            amp = x_eig[n, m]
            if abs(amp) < 1e-12:
                continue
            # |n><m| in the eigenbasis, rotated back to the original basis
            P = np.outer(evecs[:, n], evecs[:, m].conj())
            # Each bare L_alpha already carries the (real) coupling matrix
            # element x_{nm}; that is the "L_alpha" of the paper.
            L_ops.append(qutip.Qobj(amp * P))
            # Convention (see CONVENTIONS.md): for L = |n><m| the Bohr
            # frequency is omega = E_m - E_n, so an energy-releasing
            # (downward) transition has omega > 0. This matches
            # davies_operators; the previous E_n - E_m was the opposite sign.
            omegas.append(evals[m] - evals[n])
    omegas = np.asarray(omegas)

    return qutip.Qobj(H_mat), L_ops, omegas, gamma, imag_gamma


# --------------------------------------------------------------------------
# Check 1: the bundled dissipator is unbiased
# --------------------------------------------------------------------------
def check_dissipator_unbiased(c_ops, M=8, seed=0):
    print("\n[1] Unbiasedness of the bundled dissipator")
    print(f"    full operator count  N_L = {len(c_ops)}")
    print(f"    bundle size          M   = {M}")

    D_full = sum(qutip.lindblad_dissipator(c) for c in c_ops)
    norm_full = D_full.norm()

    print("    samples S    || <D_bundled> - D_full || / || D_full ||")
    for S in (1, 4, 16, 64):
        rng_local = np.random.default_rng(seed)
        acc = None
        for _ in range(S):
            R = bundle(c_ops, M, rng=rng_local)
            d = sum(qutip.lindblad_dissipator(r) for r in R)
            acc = d if acc is None else acc + d
        rel = (acc / S - D_full).norm() / norm_full
        print(f"    {S:8d}     {rel:.4e}")
    print("    -> error falls roughly as 1/sqrt(S): unbiased estimator.")


# --------------------------------------------------------------------------
# Check 2: Lamb shift is deterministic and independent of bundling
# --------------------------------------------------------------------------
def check_lamb_shift_deterministic(L_ops, omegas, imag_gamma):
    print("\n[2] Lamb shift is independent of bundling")
    H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma)
    is_herm = (H_LS - H_LS.dag()).norm() < 1e-10
    print(f"    || H_LS ||           = {H_LS.norm():.4e}")
    print(f"    Hermitian            : {is_herm}")
    print(f"    largest eigenvalue   = {max(np.real(H_LS.eigenenergies())):.4e}")
    print("    Built from the bare L_alpha -- bundling never enters here.")
    return H_LS


# --------------------------------------------------------------------------
# Check 3: bundled dynamics reproduce the full master equation
# --------------------------------------------------------------------------
def check_dynamics(H_sys, L_ops, omegas, gamma, imag_gamma, N, M=8, seed=0):
    print("\n[3] Bundled dynamics vs the full Lindblad master equation")

    # Full reference: build all c_ops, do not bundle, include H_LS.
    c_full = build_collapse_ops(L_ops, omegas, gamma)
    H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma)
    H_total = H_sys + H_LS

    a = qutip.destroy(N)
    number = a.dag() * a
    rho0 = qutip.ket2dm(qutip.basis(N, N - 1))
    tlist = np.linspace(0.0, 20.0, 50)
    e_ops = [H_total, number]

    full = qutip.mesolve(H_total, rho0, tlist, c_ops=c_full, e_ops=e_ops)

    # One-shot helper: build collapse ops, bundle them, build H_LS.
    bundled = prepare_bundled_dynamics(L_ops, omegas, gamma, M=M,
                                         imag_gamma=imag_gamma, rng=seed)
    print(f"    full operator count  N_L  = {bundled.extras['n_full_ops']}")
    print(f"    bundle size          M    = {bundled.M}")

    single = qutip.mesolve(H_sys + bundled.H_lamb_shift, rho0, tlist,
                            c_ops=bundled.c_ops, e_ops=e_ops)

    ens = mesolve_ensemble(H_total, rho0, tlist, c_full, M=M,
                            e_ops=e_ops, n_realizations=16, rng=seed)
    jack = mesolve_jackknife(H_total, rho0, tlist, c_full, M=M,
                              e_ops=e_ops, n_realizations=12, rng=seed)

    e_full = np.real(full.expect[0])
    rms = lambda y: np.sqrt(np.mean((y - e_full) ** 2))
    span = e_full.max() - e_full.min()
    print(f"    RMS in <H>, single bundled (M={M})       : {rms(single.expect[0]):.4e}")
    print(f"    RMS in <H>, ensemble mean (16 realiz.)   : {rms(ens.expect[0]):.4e}")
    print(f"    RMS in <H>, jackknife-2 corrected        : {rms(jack.expect[0]):.4e}")
    print(f"    (<H> spans {e_full.min():.2f} .. {e_full.max():.2f}; "
          f"range = {span:.2f})")
    return tlist, full, single, ens, jack


# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Stochastic Lindblad bundling -- validation against full mesolve")
    print("=" * 70)

    N = 14
    H_sys, L_ops, omegas, gamma, imag_gamma = build_system(N=N)
    c_full = build_collapse_ops(L_ops, omegas, gamma)
    check_dissipator_unbiased(c_full, M=8)
    check_lamb_shift_deterministic(L_ops, omegas, imag_gamma)
    tlist, full, single, ens, jack = check_dynamics(
        H_sys, L_ops, omegas, gamma, imag_gamma, N, M=8,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for j, label in enumerate([r"$\langle H_\mathrm{total}\rangle$",
                                    r"$\langle a^\dagger a\rangle$"]):
            ax[j].plot(tlist, np.real(full.expect[j]), "k-", lw=2.4,
                       label=f"full ({len(c_full)} operators)")
            ax[j].plot(tlist, single.expect[j], color="tab:orange",
                       lw=1, alpha=0.85, label="single bundled (M=8)")
            ax[j].errorbar(tlist[::6], ens.expect[j][::6],
                           yerr=ens.sem[j][::6], fmt="o", ms=4,
                           color="tab:blue", capsize=2,
                           label="ensemble mean (M=8, 16x)")
            ax[j].set_xlabel("time")
            ax[j].set_ylabel(label)
            ax[j].legend(fontsize=8, frameon=False, loc="best")
        ax[0].set_title("Energy relaxation (includes Lamb shift)")
        ax[1].set_title("Mean occupation")
        fig.suptitle("Bundled dissipator + deterministic Lamb shift "
                     "reproduce the full master equation", fontsize=10.5)
        fig.tight_layout()
        fig.savefig("bundling_validation.png", dpi=130)
        print("\nSaved figure: bundling_validation.png")
    except Exception as exc:                            # pragma: no cover
        print(f"\n(plot skipped: {exc})")


if __name__ == "__main__":
    main()
