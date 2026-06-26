"""Schematic of System B — anharmonic oscillator coupled to a spin (§2.4).

Draws the physical picture: an anharmonic ("hardening") oscillator whose energy
ladder has gaps that GROW toward the top, started in its highest Fock level, with
a two-level spin attached by an internal coherent coupling g.  A single ohmic
bath couples to the oscillator POSITION x only (X = x (x) I) -- it damps the
oscillator directly, and the spin relaxes only indirectly through g.

Static schematic (no computation).  Run from benchmarks/:
    python system_b_schematic.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_OSC="tab:blue"; C_SPIN="tab:purple"; C_BATH="tab:green"; C_G="tab:orange"; C_TXT="#333"

fig, ax = plt.subplots(figsize=(8.6, 4.7))
ax.set_xlim(0,10); ax.set_ylim(0,6.5); ax.axis("off")

# --- anharmonic oscillator ladder (gaps grow upward) ---
omega0, anh = 1.0, 0.1
En = np.array([omega0*(n+0.5)+anh*n*n for n in range(8)])
y0, Hl = 1.45, 3.9
ylev = y0 + (En/En[-1])*Hl
xL, xR = 1.25, 2.95
for n,y in enumerate(ylev):
    top = (n==len(ylev)-1)
    ax.plot([xL,xR],[y,y], color=C_OSC, lw=2.6 if top else 1.6,
            alpha=1.0 if top else 0.8, solid_capstyle="round", zorder=3)
ax.scatter([ (xL+xR)/2 ],[ylev[-1]], color=C_OSC, s=42, zorder=4)
ax.text((xL+xR)/2, ylev[-1]+0.18, r"start: top Fock $|N\!-\!1\rangle$",
        ha="center", va="bottom", fontsize=8.5, color=C_OSC)
# well outline (steeper-than-parabola => hardening)
wy = np.linspace(0, 1, 100)
half = 0.85*np.sqrt(wy+0.02)
ax.plot((xL+xR)/2 - half, y0+wy*Hl, color="0.7", lw=1)
ax.plot((xL+xR)/2 + half, y0+wy*Hl, color="0.7", lw=1)
ax.text((xL+xR)/2, ylev[-1]+0.62, "anharmonic oscillator\n(gaps grow upward)",
        ha="center", va="bottom", fontsize=8.5, color=C_TXT)
# cascade arrows down the ladder
for n in range(len(ylev)-1, 0, -1):
    ax.annotate("", xy=(xL-0.18, ylev[n-1]), xytext=(xL-0.18, ylev[n]),
                arrowprops=dict(arrowstyle="-|>", color=C_OSC, lw=1.1, alpha=0.55))
ax.text(xL-0.30, (ylev[0]+ylev[-1])/2, "rings\ndown", ha="right", va="center",
        fontsize=8, color=C_OSC, rotation=90)

# --- bath below, coupled to the oscillator only ---
bath = FancyBboxPatch((0.85,0.25),2.55,0.78,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        lw=1.5, edgecolor=C_BATH, facecolor="#eafaf1", zorder=1)
ax.add_patch(bath)
bx=np.linspace(1.15,3.1,150); ax.plot(bx,0.64+0.07*np.sin((bx-1.15)*7),color=C_BATH,lw=1,alpha=0.7)
for xx in np.linspace(xL+0.15,xR-0.15,4):
    ax.plot([xx,xx],[1.03,y0],color=C_BATH,ls=(0,(3,3)),lw=1.1,alpha=0.6,zorder=0)
ax.text(2.12,0.06,r"ohmic bath  $\alpha=0.3,\;k_BT=0.5,\;\omega_c=8$",
        ha="center",va="top",fontsize=8,color=C_BATH)
ax.text(3.5,1.18,r"$X=x\otimes I$"+"\n(couples to\noscillator only)",
        ha="left",va="center",fontsize=8.5,color=C_BATH)

# --- spin on the right (two levels, gap Delta) ---
xs0, xs1 = 6.7, 7.9; ysp=3.5; d=0.55
ax.plot([xs0,xs1],[ysp+d,ysp+d],color=C_SPIN,lw=2)
ax.plot([xs0,xs1],[ysp-d,ysp-d],color=C_SPIN,lw=2)
ax.annotate("",xy=(xs0-0.18,ysp+d),xytext=(xs0-0.18,ysp-d),
            arrowprops=dict(arrowstyle="<->",color=C_SPIN,lw=1.2))
ax.text(xs0-0.30,ysp,r"$\Delta$",ha="right",va="center",fontsize=10,color=C_SPIN)
ax.text((xs0+xs1)/2, ysp+d+0.22, "spin (two-level, $\\Delta=1$)",
        ha="center",va="bottom",fontsize=8.5,color=C_SPIN)
ax.text((xs0+xs1)/2, ysp-d-0.22, "no bath of its own —\nrelaxes only through $g$",
        ha="center",va="top",fontsize=8,color=C_TXT)

# --- internal coherent coupling g between oscillator and spin ---
gx=np.linspace(xR+0.05, xs0-0.05, 120)
ax.plot(gx, ysp+0.13*np.sin((gx-gx[0])*7), color=C_G, lw=2)
ax.text((xR+xs0)/2, ysp+0.45, r"$g\,(x\otimes\sigma_x)$"+"\ninternal coherent coupling",
        ha="center", va="bottom", fontsize=8.5, color=C_G)

ax.set_title(r"System B — anharmonic oscillator + spin "
             r"($\omega_0{=}1,\ \chi{=}0.1,\ \Delta{=}1,\ g{=}0.3$)",
             fontsize=11, pad=6)
fig.savefig("system_b_schematic.png", dpi=110, bbox_inches="tight")
print("wrote system_b_schematic.png")
