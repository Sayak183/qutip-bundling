import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Ultra-legibility, large-format plotting configuration for GitHub Markdown
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 2.4
plt.rcParams['xtick.major.width'] = 2.4
plt.rcParams['ytick.major.width'] = 2.4
plt.rcParams['xtick.major.size'] = 8.0
plt.rcParams['ytick.major.size'] = 8.0

LABELS = {
    # Chains (A, B)
    ("spin_chain", "energy"):       (r"$\langle H \rangle$  (total energy)",
                                     r"$\langle H \rangle$"),
    ("spin_chain", "sx"):           (r"$\sum_i\langle\sigma^x_i\rangle$  (bath coupling)",
                                     r"$\sum_i\langle\sigma^x_i\rangle$"),
    ("spin_chain", "coherence"):    ("dominant coherence",
                                     r"$\langle a|+\mathrm{h.c.}$"),

    ("mixed_chain", "energy"):      (r"$\langle H \rangle$  (total energy)",
                                     r"$\langle H \rangle$"),
    ("mixed_chain", "sx"):          (r"$\sum_i\langle\sigma^x_i\rangle$  (bath coupling)",
                                     r"$\sum_i\langle\sigma^x_i\rangle$"),
    ("mixed_chain", "sz"):          (r"$\sum_i\langle\sigma^z_i\rangle$  (magnetization)",
                                     r"$\sum_i\langle\sigma^z_i\rangle$"),
    ("mixed_chain", "coherence"):   ("dominant coherence",
                                     r"$\langle a|+\mathrm{h.c.}$"),

    # Oscillator (C)
    ("oscillator_bath", "energy"):  (r"$\langle H \rangle$  (total energy)",
                                     r"$\langle H \rangle$"),
    ("oscillator_bath", "x_sx"):    (r"$\langle x\otimes\sigma^x\rangle$  (osc-spin correlation)",
                                     r"$\langle x\otimes\sigma^x\rangle$"),
    ("oscillator_bath", "coherence"):("dominant coherence",
                                      r"$\langle a|+\mathrm{h.c.}$"),
}

def mean_curve(c):
    return np.mean(c, axis=0) if isinstance(c, (list, np.ndarray)) and np.ndim(c) == 2 else np.asarray(c)

def plot_convergence_system_huge(system_name, display_name, observables):
    filepath = f"qutip-bundling/benchmarks/data/method_comparison_{system_name}_dim64.json"
    if not os.path.exists(filepath):
        filepath = f"benchmarks/data/method_comparison_{system_name}_dim64.json"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        d = json.load(f)

    point = d['point']
    tlist_meta = d.get('meta', {}).get('tlist', {})
    t0 = tlist_meta.get('t0', 0.0)
    t1 = tlist_meta.get('t1', 5.0)
    
    ref = point['reference']
    slb = point['methods'].get('slb', [])
    obs_names = point.get('observables', [])
    
    slb.sort(key=lambda x: x['M'])
    m_values = [x['M'] for x in slb]
    colors = plt.cm.Blues(np.linspace(0.42, 1.0, len(m_values)))

    n_obs = len(observables)
    if n_obs == 4:
        # 2x2 grid for System B (16x13 inches)
        fig, axes_grid = plt.subplots(2, 2, figsize=(16.0, 13.5), dpi=300)
        axes = axes_grid.flatten()
    elif n_obs == 3:
        # 1x3 horizontal layout with tall 8.5in height (18.0 x 8.5 inches)
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 8.5), dpi=300)
    else:
        fig, axes = plt.subplots(1, n_obs, figsize=(7.0 * n_obs, 7.5), dpi=300)
        if n_obs == 1:
            axes = [axes]

    for i, obs in enumerate(observables):
        ax = axes[i]
        title_label, y_label = LABELS.get((system_name, obs), (obs, rf"$\langle {obs} \rangle$"))

        ref_curve = mean_curve(ref['curves'][obs])
        times = np.linspace(t0, t1, len(ref_curve))
        
        # Plot exact reference with thick dashed line
        ax.plot(times, ref_curve, color='black', linewidth=4.0, linestyle='--', label='Exact Reference', zorder=10)

        if obs not in obs_names:
            print(f"Observable {obs} not in {system_name}")
            continue
        obs_idx = obs_names.index(obs)

        for j, result in enumerate(slb):
            M = result['M']
            samples = np.asarray(result["samples"], dtype=float)
            c = samples[:, obs_idx, :]
            curve = np.mean(c, axis=0)
            lw = 2.6 + (j / max(1, len(slb) - 1)) * 3.4
            ax.plot(times, curve, color=colors[j], linewidth=lw, label=f'SLB (M={M})', alpha=0.92)

        ax.set_xlabel("Time $t$", fontsize=20, fontweight='bold', labelpad=12)
        ax.set_ylabel(y_label, fontsize=20, fontweight='bold', labelpad=12)
        ax.set_title(title_label, fontsize=22, fontweight='bold', pad=18)
        ax.tick_params(axis='both', which='major', labelsize=17, pad=10)
        ax.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=1.5)

        # Legend placement
        if n_obs == 4:
            if i == 1: # top right
                ax.legend(loc='upper right', framealpha=0.95, fontsize=15.5, edgecolor='lightgray')
        else:
            if i == n_obs - 1:
                ax.legend(bbox_to_anchor=(1.03, 1.0), loc='upper left', framealpha=0.95, fontsize=16, edgecolor='lightgray')

    fig.suptitle(f"{display_name} (dim 64, $N_L={point.get('n_l', '?')}$): Convergence with Bundle Size $M$",
                 fontsize=25, fontweight='heavy', y=0.99 if n_obs == 4 else 1.05)
    
    plt.tight_layout()
    out_file = f"qutip-bundling/benchmarks/convergence_dynamics_{system_name}.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"Saved {out_file} (dpi=300)")
    plt.close()

if __name__ == "__main__":
    configs = [
        ("spin_chain", "System A (TFIM Chain)", ["energy", "sx", "coherence"]),
        ("mixed_chain", "System B (Mixed Field Chain)", ["energy", "sx", "sz", "coherence"]),
        ("oscillator_bath", "System C (Oscillator Bath)", ["energy", "x_sx", "coherence"])
    ]
    for sys_name, disp_name, obs_list in configs:
        plot_convergence_system_huge(sys_name, disp_name, obs_list)
