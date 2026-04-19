# -*- coding: utf-8 -*-
"""
Modèle d'Ising - Calcul du Cumulant de Binder
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import (
    simulate_M2_M4,
    format_time
)

def binder_cumulant(M2_mean, M4_mean):
    """
    Parameters
    ----------
    M2_mean : float
    M4_mean : float

    Returns
    -------
    U_L : float
    """
    return 1.0 - M4_mean / (3.0 * M2_mean**2)

def estimate_T_c(T_s, Ls, binder_estimated, tolerance=0.01):
    """
    Parameters
    ----------
    T_s : array
        Températures
    Ls : list
        Tailles
    binder_estimated : dict
    tolerance : float
    
    Returns
    -------
    T_c : float
    """
    # On cherche où les courbes se croisent (variance minimale)
    variances = []
    for i in range(len(T_s)):
        values = [binder_estimated[L][i] for L in Ls]
        variances.append(np.var(values))
    
    # Trouver le minimum de variance
    idx_min = np.argmin(variances)
    T_c = T_s[idx_min]
    
    return T_c

def simulate_binder_cumulant(Ls, Ts, n_equilibration=10000, n_measurements=5000):
    """
    Parameters
    ----------
    Ls : list
        Tailles
    Ts : array
        Températures
    n_equilibration : int
    n_measurements : int
    
    Returns
    -------
    binder_estimated : dict
    """
    binder_estimated = {}
    
    for L in Ls:
        binder_estimated[L] = []
        for T in Ts:
            M2_mean, M4_mean = simulate_M2_M4(
                L, T, n_equilibration, n_measurements, decorrelation_sweeps=5
            )
            U_L = binder_cumulant(M2_mean, M4_mean)
            binder_estimated[L].append(U_L)
        
    for L in Ls:
        binder_estimated[L] = np.array(binder_estimated[L])
    
    return binder_estimated

if __name__ == '__main__':
    temps_init = time.time()
    
    # Paramètres
    Ls = [8, 16, 24, 32] 
    Tc_theoretical = 2.269 
    n_equilibration = 10000
    n_measurements = 5000

    # Températures autour de Tc
    T_min = 2.0
    T_max = 2.5
    steps_T = 50
    T_s = np.linspace(T_min, T_max, steps_T)

    binder_estimated = simulate_binder_cumulant(Ls, T_s, n_equilibration, n_measurements)
    T_c = estimate_T_c(T_s, Ls, binder_estimated)

    print(f"Température critique estimée: T_c ≈ {T_c:.4f}")
    print(f"Température critique théorique: T_c = {Tc_theoretical:.4f}")
    print(f"Écart relatif: {abs(T_c - Tc_theoretical)/Tc_theoretical * 100:.2f}%\n")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(Ls)))

    for i, L in enumerate(Ls):
        ax.plot(T_s, binder_estimated[L], 'o-', 
                color=colors[i], label=f'L = {L}', linewidth=2, markersize=5)
    ax.axvline(Tc_theoretical, color='red', linestyle='--', 
               linewidth=2, label=f'$T_c$ théorique = {Tc_theoretical:.3f}')
    ax.axvline(T_c, color='green', linestyle=':', 
               linewidth=2, label=f'$T_c$ estimée = {T_c:.3f}')
    ax.set_xlabel('$T$', fontsize=14)
    ax.set_ylabel('$U_L(T)$', fontsize=14)
    ax.set_title('Cumulant de Binder - $T_c$', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'../results/binder_cumulant_{T_min}_{T_max}_{steps_T}_{int(temps_init)}.pdf', 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    temps_end = time.time()
    temps_diff = temps_end - temps_init
    print(f"Temps d'exécution total: {format_time(temps_diff)}")
