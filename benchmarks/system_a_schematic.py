"""Schematic of System A — the dissipative transverse-field Ising chain.

Draws the physical picture behind §2.3 of BENCHMARKS.md: a 1-D chain of spins
with nearest-neighbour Ising coupling J and a transverse field h, all coupled to
a *single collective* ohmic bath through the global operator X = sum_i sigma^x_i.
This is a static schematic (no computation); it just renders the system so the
"one shared bath" structure is visible at a glance.

Run from the benchmarks/ folder:  python system_a_schematic.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# --- house palette (matches the tab: colours used by the benchmark figures) ---
C_SITE = "tab:blue"
C_BOND = "#3a3f44"
C_FIELD = "tab:orange"
C_BATH = "tab:green"
C_COUP = "tab:green"

fig, ax = plt.subplots(figsize=(8.2, 4.3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.axis("off")

# spin sites: three explicit, an ellipsis, then the n-th site (=> general n)
xs = [1.5, 3.2, 4.9, 6.6, 8.3]          # last-but-one slot is the ellipsis
ellipsis_idx = 3
site_xs = [x for i, x in enumerate(xs) if i != ellipsis_idx]
y_site = 3.55
r = 0.42

# nearest-neighbour Ising bonds (faded across the ellipsis to signal "continues")
for a, b in zip(xs[:-1], xs[1:]):
    crosses_gap = (a == xs[ellipsis_idx - 1]) or (b == xs[ellipsis_idx + 1])
    ax.plot([a + r, b - r], [y_site, y_site],
            color=C_BOND, lw=3, zorder=1,
            alpha=0.35 if crosses_gap else 1.0,
            ls="--" if crosses_gap else "-",
            solid_capstyle="round")
# one J label on a representative bond
ax.text((xs[0] + xs[1]) / 2, y_site + 0.22, r"$J$",
        color=C_BOND, fontsize=11, ha="center", va="bottom")

ax.text(xs[ellipsis_idx], y_site, r"$\cdots$", fontsize=16,
        ha="center", va="center", color=C_BOND)

# the bath: one wide reservoir under the whole chain
bath = FancyBboxPatch((1.0, 0.55), 7.3, 1.05,
                      boxstyle="round,pad=0.02,rounding_size=0.18",
                      linewidth=1.6, edgecolor=C_BATH,
                      facecolor="#eafaf1", zorder=1)
ax.add_patch(bath)
# a thermal wiggle inside the reservoir
wx = np.linspace(1.4, 7.9, 240)
ax.plot(wx, 1.07 + 0.10 * np.sin((wx - 1.4) * 6.0),
        color=C_BATH, lw=1.2, alpha=0.7, zorder=2)
ax.text(4.65, 0.30,
        r"collective ohmic bath:  $\gamma(\omega)=\alpha\,\omega\,e^{-|\omega|/\omega_c}/(1-e^{-\omega/k_BT})$,"
        "\n"
        r"$\alpha=0.3,\;\; k_BT=0.5,\;\; \omega_c=8$",
        fontsize=8, ha="center", va="top", color=C_BATH)

# spins + transverse field + coupling lines to the shared bath
for k, x in enumerate(site_xs):
    # dashed coupling line from each spin down into the single reservoir
    ax.plot([x, x], [y_site - r, 1.60], color=C_COUP, lw=1.3,
            ls=(0, (4, 3)), alpha=0.55, zorder=0)
    # the spin (a qubit), drawn polarised |up>
    ax.add_patch(Circle((x, y_site), r, facecolor="white",
                        edgecolor=C_SITE, lw=2.2, zorder=3))
    ax.text(x, y_site, r"$\uparrow$", fontsize=15, ha="center",
            va="center", color=C_SITE, zorder=4, fontweight="bold")
    # transverse field: a short horizontal (x-direction) arrow above the spin
    ax.add_patch(FancyArrowPatch((x - 0.30, y_site + 0.95),
                                 (x + 0.30, y_site + 0.95),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=C_FIELD, lw=1.8, zorder=3))

# site index labels (1, 2, 3, ..., n)
for x, lab in zip(site_xs, ["1", "2", "3", "n"]):
    ax.text(x, y_site - r - 0.30, lab, fontsize=8.5, ha="center",
            va="top", color="#555")

# transverse-field term label
ax.text(site_xs[-1] + 0.75, y_site + 0.95, r"$-h\,\sigma^x_i$",
        color=C_FIELD, fontsize=10, ha="left", va="center")

# coupling-operator label, tied to the bath
ax.text(8.55, 1.95, r"$X=\sum_i \sigma^x_i$" "\n" r"(one shared bath)",
        color=C_COUP, fontsize=9.5, ha="left", va="center")

# initial state + Ising term
ax.text(0.35, 5.55,
        r"$|\psi_0\rangle=|\!\uparrow\uparrow\cdots\uparrow\rangle$"
        r"$\qquad H_{\rm sys}=-J\sum_i \sigma^z_i\sigma^z_{i+1}-h\sum_i \sigma^x_i$",
        fontsize=10, ha="left", va="center", color="#222")

ax.set_title("System A — dissipative transverse-field Ising chain "
             r"($J=1.0,\ h=0.6$)", fontsize=11.5, pad=8)

fig.savefig("system_a_schematic.png", dpi=110, bbox_inches="tight")
print("wrote system_a_schematic.png")
