"""
pimd_plots.py
Usage : python pimd_plots.py
Ajuster les paramètres dans la section "PARAMÈTRES" ci-dessous.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from math import sqrt, floor
import os

# -----------------------------
# PARAMÈTRES PAR DÉFAUT
# -----------------------------
T_default = 2.0/3.0            # température (en unités réduites V0/kB)
V0_default = 5.0/3.0           # V0
a_nm = 0.04                    # paramètre 'a' (nm)
dt_default = 1e-17             # inutilisé ici
epsilon_default = 0.2         # pas de déplacement initial
n_steps_default = 20000        # nombre d'itérations Monte-Carlo
P_default = 8                  # nombre de beads / "trotters"
K_default = None               # si None, on utilisera K = P*T**2
N_REPLICATES = 10**0           # nombre de réplicats global par défaut
CONF_LEVEL = 0.95              # niveau de confiance (95%)

OUT_DIR = "figures_pimd"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# POTENTIEL DU DOUBLE PUITS
# -----------------------------
def V(x, V0=V0_default, a=1.0):
    y = x / a
    return V0 * (y**2 - 1.0)**2

# -----------------------------
# ENERGIE POTENTIELLE EFFECTIVE
# -----------------------------
def potential_energy(beads, K):
    term_V = np.mean(V(beads))
    diffs = beads - np.roll(beads, -1)
    term_spring = 0.5 * K * np.sum(diffs**2)
    return term_V + term_spring

def mean_potential_energy(beads):
    return np.mean(V(beads))

# -----------------------------
# SIMULATION METROPOLIS
# -----------------------------
def metropolis_simulation(P=None, n_steps=None, epsilon=None,
                          T=None, K=None, initial_beads=None,
                          rng=None, record_every=1):
    if P is None: P = P_default
    if n_steps is None: n_steps = n_steps_default
    if epsilon is None: epsilon = epsilon_default
    if T is None: T = T_default
    if K is None: K = P * T**2
    if rng is None: rng = np.random.default_rng()

    if initial_beads is None:
        beads = np.zeros(P)
    else:
        beads = np.array(initial_beads, dtype=float)

    beta = 1.0 / T
    centroids = []
    beads_snapshots = []
    energies = []

    accepted = 0
    trials = 0
    U_current = potential_energy(beads,K)

    for step in range(n_steps):
        print(step)
        for j in range(P):
            trials += 1
            delta = (rng.random() * 2 - 1) * epsilon
            old_pos = beads[j]
            beads[j] = old_pos + delta
            U_new = potential_energy(beads, K)
            dU = U_new - U_current
            if dU <= 0.0 or rng.random() < np.exp(-beta*dU):
                accepted += 1
                U_current = U_new
            else:
                beads[j] = old_pos

        if step % record_every == 0:
            centroids.append(np.mean(beads))
            beads_snapshots.append(beads.copy())
            energies.append(mean_potential_energy(beads))

    acceptance_rate = accepted / trials if trials > 0 else 0.0
    return {
        'centroids': np.array(centroids),
        'beads_snapshots': np.array(beads_snapshots),
        'energies': np.array(energies),
        'acceptance_rate': acceptance_rate,
        'params': {'P': P, 'n_steps': n_steps, 'epsilon': epsilon, 'T': T, 'K': K}
    }

# -----------------------------
# LANCER PLUSIEURS RÉPLIQUES
# -----------------------------
def run_ensemble(n_replicates=None, **sim_kwargs):
    if n_replicates is None: n_replicates = N_REPLICATES

    rng = np.random.default_rng()
    results = []
    for i in range(n_replicates):
        res = metropolis_simulation(rng=np.random.default_rng(rng.integers(1<<30)), **sim_kwargs)
        results.append(res)

    energies = np.array([np.mean(r['energies']) for r in results])
    acc_rates = np.array([r['acceptance_rate'] for r in results])
    beads_concat = np.vstack([r['beads_snapshots'].reshape(-1, r['params']['P']) for r in results])
    centroids_concat = np.hstack([r['centroids'] for r in results])

    mean_energy = energies.mean()
    se_energy = energies.std(ddof=1) / sqrt(len(energies))
    z = 1.96 if CONF_LEVEL == 0.95 else 1.96
    ci_energy = (mean_energy - z*se_energy, mean_energy + z*se_energy)

    return {
        'replicates': results,
        'energies_per_repl': energies,
        'acc_rates': acc_rates,
        'mean_energy': mean_energy,
        'ci_energy': ci_energy,
        'beads_concat': beads_concat,
        'centroids_concat': centroids_concat
    }

# -----------------------------
# PLOTTING UTILITIES
# -----------------------------
def plot_energy_with_acceptance(epsilon_list, P=None, T=None,
                                n_steps=None, record_every=10,
                                n_replicates=None, K=None):
    if P is None: P = P_default
    if T is None: T = T_default
    if n_steps is None: n_steps = n_steps_default
    if n_replicates is None: n_replicates = N_REPLICATES
    if K is None: K = P*T**2

    # Single replicate
    plt.figure(figsize=(9,6))
    acceptance_single = {}
    for eps in epsilon_list:
        out = metropolis_simulation(P=P, n_steps=n_steps, epsilon=eps, T=T, K=K,
                                    record_every=record_every)
        Vt = out['energies']
        acc = out['acceptance_rate']
        acceptance_single[eps] = acc
        steps = np.arange(len(Vt))*record_every
        line, = plt.plot(steps, Vt, label=f"ε={eps},acc={acc*100:.1f}%")
        # plt.text(steps[-1], Vt[-1], f"{acc*100:.1f}%", fontsize=9, color=line.get_color(), va="center")

    plt.xlabel("MC step")
    plt.ylabel("⟨V⟩ (reduced units)")
    # plt.title("Single-replicate relaxation curves")
    plt.grid(True)
    plt.legend(title="Epsilon")
    f1 = os.path.join(OUT_DIR, "energy_single_replicate.png")
    plt.savefig(f1, dpi=200)
    print("Saved:", f1)
    plt.show()

    # Ensemble
    plt.figure(figsize=(9,6))
    acceptance_ensemble = {}
    for eps in epsilon_list:
        curves = []
        acc_rates = []
        for i in range(n_replicates):
            out = metropolis_simulation(P=P, n_steps=n_steps, epsilon=eps, T=T, K=K,
                                        record_every=record_every)
            curves.append(out['energies'])
            acc_rates.append(out['acceptance_rate'])
        curves = np.array(curves)
        acceptance_ensemble[eps] = np.mean(acc_rates)
        steps = np.arange(curves.shape[1])*record_every
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        line, = plt.plot(steps, mean_curve, label=f"ε={eps}, mean_acc ={np.mean(acc_rates)*100:.1f}")
        plt.fill_between(steps, mean_curve-std_curve, mean_curve+std_curve,
                         alpha=0.25, color=line.get_color())
        # plt.text(steps[-1], mean_curve[-1], f"{np.mean(acc_rates)*100:.1f}%", fontsize=9, color=line.get_color(), va="center")

    plt.xlabel("MC step")
    plt.ylabel("⟨V⟩ (reduced units)")
    # plt.title(f"Mean ⟨V⟩(t) ± std   ({n_replicates} replicates)")
    plt.grid(True)
    plt.legend(title="Epsilon")
    f2 = os.path.join(OUT_DIR, "energy_mean_std.png")
    plt.savefig(f2, dpi=200)
    print("Saved:", f2)
    plt.show()

def plot_energy_vs_P(P_list, n_replicates=None, n_steps=None, epsilon=None, T=None):
    if n_replicates is None: n_replicates = N_REPLICATES
    if n_steps is None: n_steps = n_steps_default
    if epsilon is None: epsilon = epsilon_default
    if T is None: T = T_default

    means = []
    lowers = []
    uppers = []

    for P in P_list:
        out = run_ensemble(n_replicates, P=P, n_steps=n_steps, epsilon=epsilon, T=T, K=None)
        mean = out['mean_energy']
        ci_low, ci_up = out['ci_energy']
        means.append(mean)
        lowers.append(ci_low)
        uppers.append(ci_up)
        print(f"P={P} -> mean energy={mean:.4f}, CI=({ci_low:.4f},{ci_up:.4f})")

    P_arr = np.array(list(P_list))
    means = np.array(means)
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    plt.figure(figsize=(7,5))
    plt.fill_between(P_arr, lowers, uppers, alpha=0.2, label=f"{int(CONF_LEVEL*100)}% CI")
    plt.plot(P_arr, means, '-o', label="E pot moyen")
    plt.xlabel("Nombre de beads P")
    plt.ylabel("Énergie potentielle moyenne (unités réduites)")
    # plt.title(f"Convergence énergie vs P (n_steps={n_steps}, eps={epsilon}, replicates={n_replicates})")
    plt.grid(True)
    plt.legend()
    outpath = os.path.join(OUT_DIR, "energy_vs_P.png")
    plt.savefig(outpath, dpi=200)
    print("Saved:", outpath)
    plt.show()

# -----------------------------
# DENSITÉS ET BARRIÈRE
# -----------------------------
def plot_rho_and_barrier(beads_concat, centroids_concat,epsilon=epsilon_default, T=T_default, nbins=200, xlim=(-1.5,1.5),
                         show_beads=True, show_centroid=True, title_suffix=""):
    xs = np.linspace(xlim[0], xlim[1], 300)
    plt.figure(figsize=(10,5))

    # Densités
    plt.subplot(1,2,1)
    if show_beads and beads_concat.size>0:
        tot_data_beads=0
        data_beads = beads_concat.ravel()
        rho_beads = np.full_like(xs, 0)
        for bead in data_beads:
            if (bead>=-1.5)and(bead<=1.5):
                j=int(floor((bead+1.5)/0.01))
                rho_beads[j]+=1 
                tot_data_beads+=1
        plt.plot(xs, rho_beads/tot_data_beads, label="rho(beads)")
    if show_centroid and centroids_concat.size>0:
        rho_cent = np.full_like(xs, 0)
        tot_data_cent=0
        for cent in centroids_concat:
            if (cent>=-1.5)and(cent<=1.5):
                j=int(floor((cent+1.5)/0.01))
                rho_cent[j]+=1
                tot_data_cent+=1
        plt.plot(xs, rho_cent/tot_data_cent, label="rho(centroid)")
    plt.xlabel("x (unités réduites)")
    plt.ylabel("densité rho(x)")
    # plt.title("Densités " + title_suffix)
    plt.legend()
    plt.grid(True)

    # Barrière effective
    plt.subplot(1,2,2)
    kB_T = T
    eps = 1e-12
    if show_beads and beads_concat.size>0:
        mask = rho_beads>eps
        barrier = np.full_like(xs, np.nan)
        barrier[mask] = -kB_T*np.log(rho_beads[mask]/rho_beads[mask].max())
        plt.plot(xs, barrier, label="barrière effective (beads)")
    if show_centroid and centroids_concat.size>0:
        mask = rho_cent>eps
        barrier_c = np.full_like(xs, np.nan)
        barrier_c[mask] = -kB_T*np.log(rho_cent[mask]/rho_cent[mask].max())
        plt.plot(xs, barrier_c, label="barrière effective (centroïde)")
    plt.xlabel("x")
    plt.ylabel("-k_B T ln[rho/rho_max]")
    # plt.title("Barrière effective " + title_suffix)
    plt.legend()
    plt.grid(True)

    outpath = os.path.join(OUT_DIR, f"rho_and_barrier{('_'+title_suffix) if title_suffix else ''}.png")
    plt.savefig(outpath, dpi=200)
    print("Saved:", outpath)
    plt.show()

def plot_rho_and_barrier_ensemble(beads_concat, centroids_concat, P, T=T_default,
                                  nbins=200, xlim=(-1.5,1.5), title_suffix=""):
    xs_edges = np.linspace(xlim[0], xlim[1], nbins + 1)
    xs = 0.5 * (xs_edges[:-1] + xs_edges[1:])  # bin centers

    # Split data per replicate
    total_beads = beads_concat.shape[0]
    n_repl = total_beads // P
    samples_per_repl = beads_concat.shape[0] // n_repl

    beads_repl = beads_concat.reshape(n_repl, -1)
    cent_repl = centroids_concat.reshape(n_repl, -1)

    rho_beads_all = []
    rho_cent_all = []

    for r in range(n_repl):
        hist_beads, _ = np.histogram(beads_repl[r], bins=xs_edges, density=True)
        hist_cent, _ = np.histogram(cent_repl[r], bins=xs_edges, density=True)
        rho_beads_all.append(hist_beads)
        rho_cent_all.append(hist_cent)

    rho_beads_all = np.array(rho_beads_all)
    rho_cent_all = np.array(rho_cent_all)

    rho_beads_mean = rho_beads_all.mean(axis=0)
    rho_beads_std = rho_beads_all.std(axis=0)

    rho_cent_mean = rho_cent_all.mean(axis=0)
    rho_cent_std = rho_cent_all.std(axis=0)

    plt.figure(figsize=(10,5))

    # Left: density
    plt.subplot(1,2,1)
    plt.plot(xs, rho_beads_mean, label="mean ρ(beads)")
    plt.fill_between(xs, rho_beads_mean-rho_beads_std, rho_beads_mean+rho_beads_std, alpha=0.3)

    plt.plot(xs, rho_cent_mean, label="mean ρ(centroids)")
    plt.fill_between(xs, rho_cent_mean-rho_cent_std, rho_cent_mean+rho_cent_std, alpha=0.3)

    plt.xlabel("x")
    plt.ylabel("density ρ(x)")
    # plt.title("Mean densities ± std " + title_suffix)
    plt.legend()
    plt.grid(True)

    # Right: effective barrier
    plt.subplot(1,2,2)
    eps = 1e-12

    mask_beads = rho_beads_mean > eps
    mask_cent = rho_cent_mean > eps

    barrier_beads = np.full_like(rho_beads_mean, np.nan)
    barrier_cent  = np.full_like(rho_cent_mean,  np.nan)

    barrier_beads[mask_beads] = -T * np.log(rho_beads_mean[mask_beads]/rho_beads_mean.max())
    barrier_cent[mask_cent] = -T * np.log(rho_cent_mean[mask_cent]/rho_cent_mean.max())

    plt.plot(xs, barrier_beads, label="beads barrier")
    plt.plot(xs, barrier_cent, label="centroid barrier")

    plt.xlabel("x")
    plt.ylabel("-kB T ln(ρ/ρmax)")
    plt.title("Barrier ± std " + title_suffix)
    plt.legend()
    plt.grid(True)

    out = os.path.join(OUT_DIR, f"rho_barrier_ensemble{('_'+title_suffix) if title_suffix else ''}.png")
    plt.savefig(out, dpi=200)
    print("Saved ensemble:", out)
    plt.show()


# -----------------------------
# REPONSES AUX QUESTIONS
# -----------------------------
def question_3_find_epsilon():

    P = 1
    eps_list = [0.1,0.5,0.8,1.0]
    plot_energy_with_acceptance(eps_list, n_replicates=N_REPLICATES, P=P, n_steps=5000, T=T_default, K=None)
    # imprime suggestion
    print("Règle empirique : acceptance ~ 0.4-0.6 est souvent un bon choix.")

def question_4_energy_vs_P():

    P_list = [1,2,4,8,16,32,64]
    plot_energy_vs_P(P_list, n_replicates=N_REPLICATES, n_steps=100000, epsilon=0.8, T=T_default)

def questions_5_6_compute_rho_and_barrier():

    P = 32
    epsilon = 1.0
    n_steps = 5*10**6
    n_replicates = 5

    # First: Single simulation
    print("Running one simulation...")
    out_single = metropolis_simulation(P=P, n_steps=n_steps, epsilon=epsilon, T=T_default, K=None)
    plot_rho_and_barrier(out_single['beads_snapshots'], out_single['centroids'], 
                         epsilon, T=T_default, 
                         title_suffix=f"P{P}_T{T_default}_single")

    # Second: Ensemble mean ± std
    # print("Running ensemble...")
    # out_ens = run_ensemble(n_replicates, P=P, n_steps=n_steps, epsilon=epsilon, T=T_default, K=None)
    # plot_rho_and_barrier_ensemble(out_ens['beads_concat'], out_ens['centroids_concat'], 
    #                               P=P, T=T_default, 
    #                               title_suffix=f"P{P}_T{T_default}_ensemble")

def questions_7_8_vary_T_and_C():

    # Exemples de températures
    T_list = [0.4, 0.6]  # en unités réduites V0/kB
    P = 16
    for T in T_list:
        print(f"Run for T={T}")
        out = run_ensemble(12, P=P, n_steps=10**6, epsilon=0.5, T=T*5.0/3.0, K=None)
        plot_rho_and_barrier(out['beads_concat'], out['centroids_concat'], T=T, xlim=(-1.5,1.5),
                             title_suffix=f"P{P}_T{T}")

    # Exemples de C -> ici on illustre C=0.3 et C=0.6 en variant V0 (si on souhaite)
    # Attention : cela nécessite de reconstruire V(x) si on change V0 ; notre V() prend V0 param.
    # Pour simplicité on montre comment faire 1 run avec V0 modifié :
    for C in [0.3, 0.6]:
        V0_mod = 1.0 / C
        print(f"Example run with C={C} (V0 set to {V0_mod:.3f} in potential)")
        # Redéfinir une V locale pour ce run
        global V
        def V_local(x, V0=V0_mod, a=1.0):
            y = x / a
            return V0 * (y**2 - 1.0)**2
        V_backup = V
        V = V_local
        out = run_ensemble(10, P=16, n_steps=20000, epsilon=0.005, T=0.4*V0_mod, K=None)
        plot_rho_and_barrier(out['beads_concat'], out['centroids_concat'], T=0.4,
                             title_suffix=f"C{C}")
        # restore V
        V = V_backup


if __name__ == "__main__":
    print("Started. Saves figures in directory", OUT_DIR)


    # question_3_find_epsilon()

    #question_4_energy_vs_P()

    questions_5_6_compute_rho_and_barrier()

    # questions_7_8_vary_T_and_C()

    print("Finished")