# -*- coding: utf-8 -*-
"""
Modèle d'Ising - Étude des exposants critiques
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.utils import (
    simulate_single_temperature,
)

def power_law(x, a, b):
    """
    Parameters
    ----------
    x : list
        Variable indépendante
    a : float
        Coefficient
    b : float
        Exposant
    Returns
    -------
    y : list
    """
    return a * x**b

if __name__ == '__main__':
    time_init = time.time()
    
    print('Started...')
    
    # Paramètres
    Tc = 2.269
    Ls = np.array([4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192])
    n_equilibration = 10000
    n_measurements = 5000
    
    # Valeurs théoriques des exposants critiques (2D Ising)
    # β = 1/8, γ = 7/4, ν = 1
    gamma_nu_theorique = 7/4  # = 1.75
    beta_nu_theorique = -1/8  # = -0.125 (négatif car M ∝ L^(-β/ν))
    
    magnetizations = []
    susceptibilities = []
    
    for i, L in enumerate(Ls):
        time_L_init = time.time()
        if L <= 32:
            n_equilibration = 10000
            n_measurements = 5000
        elif L <= 64:
            n_equilibration = 50000
            n_measurements = 20000
        else:
            n_equilibration = 100000
            n_measurements = 50000
        print(f"L={L} avec n_equilibration={n_equilibration} et n_measurements={n_measurements}...")
        _, m, _, chi = simulate_single_temperature(L, Tc, n_equilibration, n_measurements, decorrelation_sweeps=10)
        magnetizations.append(m)
        susceptibilities.append(chi)
        time_L_end = time.time()
        print(f"Temps de calcul L={L} : {time_L_end - time_L_init:.2f} secondes \n")
    
    magnetizations = np.array(magnetizations)
    susceptibilities = np.array(susceptibilities)
    
    print("Fitting...")
    
    # T - T_c = 0
    # χ(T_c, L) = L^(γ/ν) * F_χ [0]
    # Fit susceptibilité en log-log: log(χ) = const + (γ/ν) log(L)
    chi_polyfit, chi_covariance = np.polyfit(np.log(Ls), np.log(susceptibilities), 1, cov=True)
    gamma_nu_estimation = chi_polyfit[0]
    gamma_error = np.sqrt(chi_covariance[0, 0])
    chi_params = np.array([np.exp(chi_polyfit[1]), gamma_nu_estimation])

    # Fit aimantation: M(L) ∝ L^(-β/ν)
    m_polyfit, m_covariance = np.polyfit(np.log(Ls), np.log(magnetizations), 1, cov=True)
    beta_nu_estimation = m_polyfit[0]
    beta_error = np.sqrt(m_covariance[0, 0])
    m_params = np.array([np.exp(m_polyfit[1]), beta_nu_estimation])
    
    print(f"Exposant critique estimé γ/ν = {gamma_nu_estimation:.3f} ± {gamma_error:.3f} (théorique: {gamma_nu_theorique:.3f})")
    print(f"Exposant critique estimé -β/ν = {beta_nu_estimation:.3f} ± {beta_error:.3f} (théorique: {beta_nu_theorique:.3f})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Susceptibilité χ(L) linéaire
    ax = axes[0, 0]
    ax.plot(Ls, susceptibilities, 'o', color='darkblue', markersize=8, label='Données')
    if chi_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        chi_fit = power_law(L_fit, *chi_params)
        ax.plot(L_fit, chi_fit, 'r--', linewidth=2, 
                label=f'Fit: ${chi_params[0]:.2f} \\times L^{{{gamma_nu_estimation:.3f}}}$')
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('$\\chi(T_c, L)$', fontsize=12)
    ax.set_title(f'Susceptibilité à $T_c = {Tc}$', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Susceptibilité χ(L) log-log
    ax = axes[0, 1]
    ax.loglog(Ls, susceptibilities, 'o', color='darkblue', markersize=8, label='Données')
    if chi_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        chi_fit = power_law(L_fit, *chi_params)
        ax.loglog(L_fit, chi_fit, 'r--', linewidth=2, 
                  label=f'Fit: $\\gamma/\\nu = {gamma_nu_estimation:.3f} \\pm {gamma_error:.3f}$')
        chi_theorique = chi_params[0] * L_fit**(gamma_nu_theorique)
        ax.loglog(L_fit, chi_theorique, 'g:', linewidth=2, 
                  label=f'Théorique: $\\gamma/\\nu = {gamma_nu_theorique:.3f}$')
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('$\\chi(T_c, L)$', fontsize=12)
    ax.set_title('Susceptibilité', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # 3. Aimantation M(L) linéaire
    ax = axes[1, 0]
    ax.plot(Ls, magnetizations, 'o', color='darkred', markersize=8, label='Données')
    if m_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        m_fit = power_law(L_fit, *m_params)
        ax.plot(L_fit, m_fit, 'b--', linewidth=2, 
                label=f'Fit: ${m_params[0]:.3f} \\times L^{{{beta_nu_estimation:.3f}}}$')
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('$\\langle |m| \\rangle (T_c, L)$', fontsize=12)
    ax.set_title(f'Aimantation à $T_c = {Tc}$', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. Aimantation M(L) log-log
    ax = axes[1, 1]
    ax.loglog(Ls, magnetizations, 'o', color='darkred', markersize=8, label='Données')
    if m_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        m_fit = power_law(L_fit, *m_params)
        ax.loglog(L_fit, m_fit, 'b--', linewidth=2, 
                  label=f'Fit: $-\\beta/\\nu = {beta_nu_estimation:.3f} \\pm {beta_error:.3f}$')
        m_theorique = m_params[0] * L_fit**(beta_nu_theorique)
        ax.loglog(L_fit, m_theorique, 'g:', linewidth=2, 
                  label=f'Théorique: $-\\beta/\\nu = {beta_nu_theorique:.3f}$')
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('$\\langle |m| \\rangle (T_c, L)$', fontsize=12)
    ax.set_title('Aimantation', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'../results/exposants_critiques_{int(time_init)}.pdf', 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    time_end = time.time()
    print('Finished')
    print(f"Temps total d'exécution : {time_end - time_init:.2f} secondes")
    