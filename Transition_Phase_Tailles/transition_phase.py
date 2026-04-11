# -*- coding: utf-8 -*-
"""
Modèle d'Ising - Étude de la transition de phase
Influence de la taille du système
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import (
    remplissage_aleatoire_reseau,
    calculate_magnetization,
    theoretical_magnetization,
    calculate_total_energy,
    MonteCarlo,
    format_time
)

def simulate_temperature(L, T, n_equilibration=1000, n_measurements=1000, J=1.0, h=0.0):
    """
    Parameters
    ----------
    L : int
        Taille du réseau
    T : float
        Température
    n_equilibration : int
        Nombre de pas Monte Carlo pour l'équilibration
    n_measurements : int
        Nombre de mesures pour les moyennes
    J : float
        Constante d'échange
    h : float
        Champ magnétique appliqué
    Returns
    -------
    results : dict
    """
    N = L**2
    Reseau = remplissage_aleatoire_reseau(L)
    
    # Équilibration
    for _ in range(n_equilibration):
        Reseau = MonteCarlo(N, L, Reseau, T, J, h)
    
    energies = []
    magnetizations = []
    
    for _ in range(n_measurements):
        Reseau = MonteCarlo(N, L, Reseau, T, J, h)
        E = calculate_total_energy(Reseau, J, h)
        M = calculate_magnetization(Reseau)
        energies.append(E)
        magnetizations.append(np.abs(M))  # On prend la valeur absolue
    
    energies = np.array(energies)
    magnetizations = np.array(magnetizations)
    
    # Calcul des moyennes et fluctuations
    E_mean = np.mean(energies)
    E2_mean = np.mean(energies**2)
    M_mean = np.mean(magnetizations)
    M2_mean = np.mean(magnetizations**2)
    
    # Énergie par spin
    e = E_mean / N
    
    # Aimantation par spin
    m = M_mean / N
    
    # Susceptibilité magnétique χ(T)
    chi = (N / T) * (M2_mean - M_mean**2) 

    # Chaleur spécifique  C(T)
    C = (1 / (T**2)) * (E2_mean - E_mean**2)
    
    return {
        'energie_per_spin': e,
        'magnetization_per_spin': m,
        'specific_heat': C,
        'susceptibility': chi
    }


if __name__ == '__main__':
    temps_init = time.time()
    
    # Paramètres
    Ls = [4, 8, 16, 32, 64]  # Tailles du système
    Tc = 2.269  # Température critique théorique

    # Températures autour de Tc
    T_s = np.linspace(1.0, 4.0, 20)

    # Paramètres de simulation
    n_equilibration = 10000   # Pas d'équilibration
    n_measurements = 1000    # Nombre de mesures

    # Stockage
    results = {}

    for L in Ls:
        print(f"L = {L}...")
        results[L] = {
            'T_s': T_s,
            'energie': [],
            'magnetization': [],
            'specific_heat': [],
            'susceptibility': []
        }
        
        for i, T in enumerate(T_s):
            res = simulate_temperature(L, T, n_equilibration, n_measurements)
            results[L]['energie'].append(res['energie_per_spin'])
            results[L]['magnetization'].append(res['magnetization_per_spin'])
            results[L]['specific_heat'].append(res['specific_heat'])
            results[L]['susceptibility'].append(res['susceptibility'])
            
            if (i+1) % 5 == 0:
                print(f"  progress: {i+1}/{len(T_s)} températures")

    # Conversion en arrays numpy
    for L in Ls:
        results[L]['energie'] = np.array(results[L]['energie'])
        results[L]['magnetization'] = np.array(results[L]['magnetization'])
        results[L]['specific_heat'] = np.array(results[L]['specific_heat'])
        results[L]['susceptibility'] = np.array(results[L]['susceptibility'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(Ls))) # [violet, vert, jaune, ...]

    # 1. Énergie par spin
    ax = axes[0, 0]
    for i, L in enumerate(Ls):
        ax.plot(T_s, results[L]['energie'], 'o-', 
                color=colors[i], label=f'L = {L}', linewidth=1.5, markersize=4)
    ax.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
    ax.set_xlabel('T', fontsize=12)
    ax.set_ylabel('$\\langle e \\rangle$', fontsize=12)
    ax.set_title('Énergie par spin', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Aimantation par spin
    ax = axes[0, 1]
    for i, L in enumerate(Ls):
        ax.plot(T_s, results[L]['magnetization'], 'o-', 
                color=colors[i], label=f'L = {L}', linewidth=1.5, markersize=4)
    # Courbe théorique
    m_theo = theoretical_magnetization(T_s, Tc)
    ax.plot(T_s, m_theo, 'k--', linewidth=2.5, label='Théorie')
    ax.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
    ax.set_xlabel('T', fontsize=12)
    ax.set_ylabel('$\\langle |m| \\rangle$', fontsize=12)
    ax.set_title('Aimantation par spin', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Chaleur spécifique
    ax = axes[1, 0]
    for i, L in enumerate(Ls):
        ax.plot(T_s, results[L]['specific_heat'], 'o-', 
                color=colors[i], label=f'L = {L}', linewidth=1.5, markersize=4)
    ax.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
    ax.set_xlabel('T', fontsize=12)
    ax.set_ylabel('$C$', fontsize=12)
    ax.set_title('Chaleur spécifique', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Susceptibilité magnétique
    ax = axes[1, 1]
    for i, L in enumerate(Ls):
        ax.plot(T_s, results[L]['susceptibility'], 'o-', 
                color=colors[i], label=f'L = {L}', linewidth=1.5, markersize=4)
    ax.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
    ax.set_xlabel('T', fontsize=12)
    ax.set_ylabel('$\\chi$', fontsize=12)
    ax.set_title('Susceptibilité magnétique', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/transition_phase_tailles.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    temps_end = time.time()
    temps_diff = temps_end - temps_init
    print(f"Temps d'exécution : {format_time(temps_diff)}")

