import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from common import add_settings_footer

# =====================================================================
# Configuration & Toggles
# =====================================================================
SYSTEMS = ["spin_chain", "mixed_chain", "oscillator_bath"]
SYSTEM_TITLES = {
    "spin_chain": "System A - TFIM chain",
    "mixed_chain": "System B - mixed-field chain",
    "oscillator_bath": "System C - oscillator + spin",
}
DIMS = [16, 32, 64]
COLORS = {16: '#1f77b4', 32: '#ff7f0e', 64: '#2ca02c'}
MARKERS = {16: 'o', 32: 's', 64: '^'}

# Toggle: True = evaluate at the worst-time slice t* (standard R1 anatomy)
#         False = use the overall time-averaged error across the full trajectory
EVALUATE_AT_WORST_TIME = True

# Bundle (M) Range Toggles
MIN_M = 2
MAX_M = 32

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_r1_data(system, dim):
    path = DATA_DIR / f"accuracy_vs_M_{system}_dim{dim}.json"
    if not path.exists():
        raise FileNotFoundError(f"No Result 1 JSON data found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def calculate_exponent(M_array, bias_array, sem_array):
    M_valid = []
    bias_valid = []
    
    # Strict quoting rule: >=3 points must clear 2x SEM
    for m, b, s in zip(M_array, bias_array, sem_array):
        if b > 2 * s:
            M_valid.append(m)
            bias_valid.append(b)
            
    if len(M_valid) >= 3:
        slope, intercept, r_value, p_value, std_err = linregress(
            np.log10(M_valid), np.log10(bias_valid)
        )
        return slope, True
    else:
        return None, False

def calculate_sem_exponent(M_array, sem_array):
    if len(M_array) >= 3:
        slope, intercept, r_value, p_value, std_err = linregress(
            np.log10(M_array), np.log10(sem_array)
        )
        return slope, True
    else:
        return None, False

# =====================================================================
# Plotting Logic
# =====================================================================
for system in SYSTEMS:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    sys_title = SYSTEM_TITLES.get(system, system.replace('_', ' ').title())
    
    for dim in DIMS:
        try:
            data = load_r1_data(system, dim)
        except FileNotFoundError as e:
            print(f"[!] {e}")
            continue

        # ---------------------------------------------------------
        # Anatomy Calculation from Raw Time-Series Samples
        # ---------------------------------------------------------
        try:
            sweep_data = data['slb_sweep']
            ref_arr = np.array(data['reference_energy'])
            
            M_vals, bias_vals, sem_vals = [], [], []
            realizations = 0
            
            # Find the worst-time index t* from the smallest M bundle if toggled
            worst_t_idx = 0
            if EVALUATE_AT_WORST_TIME and len(sweep_data) > 0:
                smallest_item = sweep_data[0] 
                s_arr = np.array(smallest_item['samples_energy'])
                if s_arr.ndim == 1:
                    s_arr = s_arr.reshape(1, -1)
                smallest_mean = np.mean(s_arr, axis=0)
                worst_t_idx = np.argmax(np.abs(smallest_mean - ref_arr))

            for item in sweep_data:
                m_val = item.get('M', item.get('bundles'))
                
                samples_arr = np.array(item['samples_energy'])
                if samples_arr.ndim == 1:
                    samples_arr = samples_arr.reshape(1, -1)
                    
                # Use every realization the run saved. Capping at 16 threw
                # away 92% of the data and inflated the SEM by sqrt(200/16) =
                # 3.5x, which pushed points below the bias > 2*SEM bar and left
                # curves labelled "Level/Marginal" that in fact resolve cleanly.
                # With all 200 every point certifies and the fitted exponents
                # tighten onto the theoretical -1 (e.g. spin chain dim 32 moves
                # from -0.82 to -0.97). The neighbouring decomposition figure
                # already used all of them, so this also makes the two agree.
                    
                n_acc = samples_arr.shape[0]
                realizations = n_acc
                
                if EVALUATE_AT_WORST_TIME:
                    run_slice = samples_arr[:, worst_t_idx] 
                    mean_val = np.mean(run_slice)
                    ref_val = ref_arr[worst_t_idx]
                    
                    bias = np.abs(mean_val - ref_val)
                    if n_acc > 1:
                        sem = np.std(run_slice, ddof=1) / np.sqrt(n_acc)
                    else:
                        sem = 0.0
                else:
                    mean_traj = np.mean(samples_arr, axis=0)
                    mse = np.mean((mean_traj - ref_arr)**2)
                    if n_acc > 1:
                        sem_sq = np.mean(np.var(samples_arr, axis=0, ddof=1)) / n_acc
                    else:
                        sem_sq = 0.0
                    bias_sq = max(0.0, mse - sem_sq)
                    bias = np.sqrt(bias_sq)
                    sem = np.sqrt(sem_sq)

                M_vals.append(m_val)
                bias_vals.append(bias)
                sem_vals.append(sem)

            M_vals = np.array(M_vals)
            bias = np.array(bias_vals)
            sem = np.array(sem_vals)

            # Filter M range using the configuration toggles at the top
            valid_mask = (M_vals >= MIN_M) & (M_vals <= MAX_M)
            M_vals = M_vals[valid_mask]
            bias = bias[valid_mask]
            sem = sem[valid_mask]

        except KeyError as e:
            print(f"\n[!] {e} not found for {system} (dim={dim}).")
            continue 

        # ---------------------------------------------------------
        # Visual Layers with Exponents for both Bias and SEM
        # ---------------------------------------------------------
        bias_exp, bias_cert = calculate_exponent(M_vals, bias, sem)
        bias_suffix = rf" ($\propto M^{{{bias_exp:.2f}}}$)" if bias_cert else " (Level/Marginal)"
        
        sem_exp, sem_cert = calculate_sem_exponent(M_vals, sem)
        sem_suffix = rf" ($\propto M^{{{sem_exp:.2f}}}$)" if sem_cert else ""
        
        # Bias (Solid Line with Markers)
        ax.plot(M_vals, bias, color=COLORS[dim], linestyle='-', linewidth=2, 
                marker=MARKERS[dim], markersize=6,
                label=f"Dim {dim} Bias{bias_suffix}")
        
        # SEM Line (Dashed Line with Markers)
        ax.plot(M_vals, sem, color=COLORS[dim], linestyle='--', linewidth=1.5, 
                marker=MARKERS[dim], markersize=5, alpha=0.85,
                label=rf"Dim {dim} SEM{sem_suffix} ($N_r={realizations}$)")

    # =====================================================================
    # Formatting & Output
    # =====================================================================
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_xlabel('Number of Bundles ($M$)', fontsize=12)
    
    y_label_text = 'Error of Energy (at worst time $t^*$)' if EVALUATE_AT_WORST_TIME else 'Time-Averaged Error of Energy'
    ax.set_ylabel(y_label_text, fontsize=12)
    
    ax.set_title(f'Size Invariance: {sys_title}', fontsize=14, pad=12)

    # Place legend outside to avoid obscuring data lines
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # Add suite-style settings footer caption
    footer_text = (
        f"t* = argmax of error of the lowest-M estimate, held for all M; "
        f"uniform {realizations} realizations at every M | "
        rf"expected scaling: bias $\propto 1/M$, SEM $\propto 1/\sqrt{{M}}$"
    )
    add_settings_footer(fig, footer_text, fontsize=10, wrap_chars=130)

    plt.tight_layout()
    
    # Save as PNG
    output_filename = f"accuracy_vs_M_invariance_{system}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved invariance plot to {output_filename}")
    
    plt.close(fig)
