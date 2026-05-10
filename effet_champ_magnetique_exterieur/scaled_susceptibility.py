#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle d'Ising, Scaled susceptibility, méthode de Monte Carlo

@author: carguello
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
output_dir = Path(__file__).parent.parent / "results" / "effet-champ-magnetique-exterieur"

import time
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import (
    remplissage_aleatoire_reseau,
    equilibrate_system,
    collect_measurements,
    theoretical_magnetization,
    calculate_susceptibility,
    format_time
)

def simulate_scaled_susceptibility_at_x(L, T, J, x, d=2, n_runs=20, n_equilibration=20000, n_measurements=5000, decorrelation_sweeps=2, verbose=False):
    """
    Parameters
    ----------
    L : int
    T : float
    J : float
    x : float
        Variable réduite : x = H*L^d/J
    d : int
        Dimension spatiale (2 pour Ising 2D)
    n_runs : int - Nombre de runs indépendants
    n_equilibration : int
    n_measurements : int
    decorrelation_sweeps : int
    verbose : bool
        Si True, affiche la progression des runs
    
    Returns
    -------
    chi_scaled : float : Susceptibilité scalée χ_L / L^d
    """
    h = x * J / (L**d)
    N = L * L
    
    m_all_measurements = []
    
    t_start = time.time()
    for run in range(n_runs):
        Reseau = remplissage_aleatoire_reseau(L)
        Reseau = equilibrate_system(Reseau, L, T, J, h, n_equilibration)
        Reseau, magnetizations, _ = collect_measurements(
            Reseau, L, T, J, h, 
            n_measurements, 
            decorrelation_sweeps, 
            measure_energy=False
        )
        m_all_measurements.extend(magnetizations)
        
        if verbose and (run + 1) % 5 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (run + 1) * (n_runs - run - 1)
            print(f"    Run {run+1}/{n_runs} terminé - Temps écoulé: {format_time(elapsed)} - ETA: {format_time(eta)}")
    
    # Calculer χ sur l'ensemble global
    m_all_measurements = np.array(m_all_measurements)
    m_global = np.mean(m_all_measurements)
    m2_global = np.mean(m_all_measurements**2)
    chi = calculate_susceptibility(N, T, m2_global, m_global)
    chi_scaled = chi / (L**d)
    
    return chi_scaled

if __name__ == "__main__":
    temps_init = time.time()
    
    T = 2.1
    J = 1.0
    d = 2
    Lvalues = np.array([4, 6, 8, 10, 12, 14])
    x_scaled = np.linspace(0, 8, 41)

    Msp = float(theoretical_magnetization(np.array([T]), J=J)[0])
    Beta = 1/T
    # Simulations independants
    n_runs = 20

    plt.figure(figsize=(7, 5))

    for idx, L in enumerate(Lvalues):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(Lvalues)}] Simulation pour L = {L}")
        print(f"{'='*60}")
        t_start_L = time.time()
        chi_scaled = []

        for i, x in enumerate(x_scaled):
            t_x_start = time.time()
            print(f"  x = {x:.2f} [{i+1}/{len(x_scaled)}]")
            chi_scaled_value = simulate_scaled_susceptibility_at_x(
                L, T, J, x, d=d, 
                n_runs=n_runs, 
                n_equilibration=100000,  # 1×10^5 PMC
                n_measurements=700000,   # 7×10^5 mesures
                decorrelation_sweeps=2,  # Δτ = 2 PMC → τ_prod = 1.4×10^6 PMC
                verbose=True             # Afficher la progression des runs
                # Total: 1.5×10^6 PMC par réplique (comme Binder-Landau 1984)
            )
            chi_scaled.append(chi_scaled_value)
            t_x_end = time.time()
            
            # Estimation du temps restant
            if i > 0:
                avg_time_per_x = (time.time() - t_start_L) / (i + 1)
                eta_L = avg_time_per_x * (len(x_scaled) - i - 1)
                print(f"  ✓ x = {x:.2f} terminé en {format_time(t_x_end - t_x_start)} | ETA pour L={L}: {format_time(eta_L)}\n")
        
        t_end_L = time.time()
        print(f"\n{'='*60}")
        print(f"L = {L} TERMINÉ en {format_time(t_end_L - t_start_L)}")
        print(f"{'='*60}\n")
        plt.scatter(x_scaled, chi_scaled, s=30, alpha=0.7, label=f"L={L}")

    # Model
    x_theory = np.linspace(0, 8, 300)
    chi_theory = (Msp**2 * Beta) / np.cosh(x_theory * Msp * Beta)**2
    plt.plot(x_theory, chi_theory, 'k--', linewidth=2, label='Binder-Landau')
    plt.xlabel(r"$HL^2/J$")
    plt.ylabel(r"$\chi_L/L^2$")
    plt.title(r"Susceptibilité réduite, $T=2.1$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"scaled_susceptibility_monte_carlo_T={T}.pdf", format="pdf", bbox_inches='tight')
    plt.show()

    temps_end = time.time()
    temps_total = temps_end - temps_init
    
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLÈTE")
    print(f"{'='*60}")
    print(f"Temps total d'exécution : {format_time(temps_total)}")
    print(f"Tailles simulées : {Lvalues}")
    print(f"Points par taille : {len(x_scaled)}")
    print(f"PMC par réplique : 1.5×10⁶")
    print(f"Répliques par point : {n_runs}")
    print(f"{'='*60}\n")
