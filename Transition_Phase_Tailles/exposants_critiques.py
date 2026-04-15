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
from numba import njit
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
    temps_init = time.time()
    
    # Paramètres
    Tc = 2.269
    Ls = np.array([8, 16, 32, 64, 128])
    
    # Valeurs théoriques
    # β = 1/8, γ = 7/4, ν = 1
    beta_theo = 1/8
    gamma_theo = 7/4
    nu_theo = 1.0
    gamma_over_nu_theo = gamma_theo / nu_theo
    beta_over_nu_theo = beta_theo / nu_theo

    # Paramètres de simulation
    n_equilibration = 10000   # Équilibration à T_c
    n_measurements = 5000     # Mesures
    
    # Stockage des résultats
    magnetizations = []
    susceptibilities = []
    
    for i, L in enumerate(Ls):
        _, m, _, chi = simulate_single_temperature(L, Tc, n_equilibration, n_measurements)
        magnetizations.append(m)
        susceptibilities.append(chi)
    
    magnetizations = np.array(magnetizations)
    susceptibilities = np.array(susceptibilities)
    
    # Fit susceptibilité: χ(L) ∝ L^(γ/ν)
    chi_params, chi_covariance = curve_fit(power_law, Ls, susceptibilities, p0=[1.0, 1.0])
    chi_perr = np.sqrt(np.diag(chi_covariance))
    gamma_nu_estimation = chi_params[1]
    gamma_error = chi_perr[1]

    # Fit aimantation: M(L) ∝ L^(-β/ν)
    m_params, m_covariance = curve_fit(power_law, Ls, magnetizations, p0=[1.0, 1.0])
    m_perr = np.sqrt(np.diag(m_covariance))
    beta_nu_estimation = m_params[1]
    beta_error = m_perr[1]
    
    # beta_over_nu = -minus_beta_over_nu if minus_beta_over_nu is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Susceptibilité χ(L) en échelle linéaire
    ax = axes[0, 0]
    ax.plot(Ls, susceptibilities, 'o', color='darkblue', markersize=8, label='Données')
    if chi_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        chi_fit = power_law(L_fit, *chi_params)
        ax.plot(L_fit, chi_fit, 'r--', linewidth=2, 
                label=f'Fit: ${chi_params[0]:.2f} \\times L^{{{gamma_nu_estimation:.3f}}}$')
    ax.set_xlabel('Taille du système L', fontsize=12)
    ax.set_ylabel('$\\chi(T_c, L)$', fontsize=12)
    ax.set_title(f'Susceptibilité à $T_c = {Tc}$', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Susceptibilité χ(L) en échelle log-log
    ax = axes[0, 1]
    ax.loglog(Ls, susceptibilities, 'o', color='darkblue', markersize=8, label='Données')
    if chi_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        chi_fit = power_law(L_fit, *chi_params)
        ax.loglog(L_fit, chi_fit, 'r--', linewidth=2, 
                  label=f'$\\gamma/\\nu = {gamma_nu_estimation:.3f} \\pm {gamma_error:.3f}$')
        # Ligne théorique
        chi_theo = chi_params[0] * Ls**(gamma_over_nu_theo)
        ax.loglog(Ls, chi_theo, 'g:', linewidth=2, 
                  label=f'Théorie: $\\gamma/\\nu = {gamma_over_nu_theo:.3f}$')
    ax.set_xlabel('Taille du système L', fontsize=12)
    ax.set_ylabel('$\\chi(T_c, L)$', fontsize=12)
    ax.set_title('Susceptibilité (échelle log-log)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # 3. Aimantation M(L) en échelle linéaire
    ax = axes[1, 0]
    ax.plot(Ls, magnetizations, 'o', color='darkred', markersize=8, label='Données')
    if m_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        m_fit = power_law(L_fit, *m_params)
        ax.plot(L_fit, m_fit, 'b--', linewidth=2, 
                label=f'Fit: ${m_params[0]:.3f} \\times L^{{{beta_nu_estimation:.3f}}}$')
    ax.set_xlabel('Taille du système L', fontsize=12)
    ax.set_ylabel('$\\langle |m| \\rangle (T_c, L)$', fontsize=12)
    ax.set_title(f'Aimantation à $T_c = {Tc}$', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. Aimantation M(L) en échelle log-log
    ax = axes[1, 1]
    ax.loglog(Ls, magnetizations, 'o', color='darkred', markersize=8, label='Données')
    if m_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        m_fit = power_law(L_fit, *m_params)
        ax.loglog(L_fit, m_fit, 'b--', linewidth=2, 
                  label=f'$-\\beta/\\nu = {beta_nu_estimation:.3f} \\pm {beta_error:.3f}$')
        # Ligne théorique
        m_theo = m_params[0] * Ls**(-beta_over_nu_theo)
        ax.loglog(Ls, m_theo, 'g:', linewidth=2, 
                  label=f'Théorie: $-\\beta/\\nu = {-beta_over_nu_theo:.3f}$')
    ax.set_xlabel('Taille du système L', fontsize=12)
    ax.set_ylabel('$\\langle |m| \\rangle (T_c, L)$', fontsize=12)
    ax.set_title('Aimantation (échelle log-log)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'../results/exposants_critiques_{int(temps_init)}.pdf', 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.show()