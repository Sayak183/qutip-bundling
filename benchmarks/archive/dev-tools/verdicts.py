import json, glob, numpy as np
for p in sorted(glob.glob("convergence_progress_*_dim*.json")):
    d = json.load(open(p))
    M = np.array(d["M"]); stat = np.array(d["stat"])
    bias = np.array(d["bias"]); bjk = np.array(d["bias_jk"]); n = d["n_real"]
    sem = stat/np.sqrt(n); above = bjk > sem
    print(f"\n{d['system']} dim {d['dim']} ({n} realizations)")
    print("  M:", M.tolist())
    print("  bias    :", [f"{b:.2e}" for b in bias])
    print("  bias_jk :", [f"{b:.2e}" for b in bjk])
    s_un = np.polyfit(np.log(M), np.log(bias), 1)[0]
    if above.sum() >= 2:
        s_jk = np.polyfit(np.log(M[above]), np.log(bjk[above]), 1)[0]
        print(f"  VERDICT OK: bias rate {s_un:.2f} -> {s_jk:.2f} "
              f"({int(above.sum())} pts above floor, max reduction {np.nanmax(bias/bjk):.1f}x)")
    else:
        print(f"  VERDICT noise-limited: {int(above.sum())} pts above floor "
              f"(uncorrected {s_un:.2f}); max reduction {np.nanmax(bias/bjk):.1f}x")
