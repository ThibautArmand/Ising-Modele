# -*- coding: utf-8 -*-
"""
Modèle d'Ising - Étude des exposants critiques
Analyse du finite-size scaling à T = T_c

Objectif:
- Démontrer que χ(T_c, L) ∝ L^(γ/ν)
- Démontrer que M(T_c, L) ∝ L^(-β/ν)
- Calculer les exposants critiques γ/ν et β/ν
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
    remplissage_aleatoire_reseau,
    calculate_magnetization,
    MonteCarlo,
    format_time
)

@njit
def simulate_at_Tc(L, T_c, n_equilibration=10000, n_measurements=5000, J=1.0, h=0.0):
    """
    Parameters
    ----------
    L : int
        Taille du réseau
    T_c : float
        Température critique
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
    tuple : (m_mean, chi)
        m_mean : float - Aimantation moyenne par spin
        chi : float - Susceptibilité magnétique
    """
    N = L**2
    Reseau = remplissage_aleatoire_reseau(L)
    
    # Équilibration - plus longue car on est à T_c
    for _ in range(n_equilibration):
        Reseau = MonteCarlo(N, L, Reseau, T_c, J, h)
    
    magnetizations = np.empty(n_measurements, dtype=np.float64)
    
    for i in range(n_measurements):
        Reseau = MonteCarlo(N, L, Reseau, T_c, J, h)
        magnetizations[i] = np.abs(calculate_magnetization(Reseau))
    
    # Calcul des moyennes et fluctuations
    M_mean = np.mean(magnetizations)
    M2_mean = np.mean(magnetizations**2)
    
    # Aimantation par spin
    m_mean = M_mean / N
    
    # Susceptibilité magnétique χ(T_c)
    chi = (N / T_c) * (M2_mean - M_mean**2)
    
    return (m_mean, chi)


def power_law(x, a, b):
    """
    Loi de puissance: y = a * x^b
    
    Parameters
    ----------
    x : array
        Variable indépendante
    a : float
        Coefficient
    b : float
        Exposant
    
    Returns
    -------
    y : array
    """
    return a * x**b


def fit_power_law(L_values, data, data_name=""):
    """
    Parameters
    ----------
    L_values : array
        Tailles du système
    data : array
        Données à fitter
    data_name : str
        Nom de la grandeur pour l'affichage
    
    Returns
    -------
    tuple : (params, covariance, exponent, error)
        params : array - Paramètres du fit [a, b]
        covariance : array - Matrice de covariance
        exponent : float - Exposant b
        error : float - Erreur sur l'exposant
    """
    try:
        params, covariance = curve_fit(power_law, L_values, data, p0=[1.0, 1.0])
        perr = np.sqrt(np.diag(covariance))
        exponent = params[1]
        error = perr[1]
        
        return params, covariance, exponent, error
    except Exception as e:
        print(f"Erreur lors du fit de {data_name}: {e}")
        return None, None, None, None


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
        m, chi = simulate_at_Tc(L, Tc, n_equilibration, n_measurements)
        magnetizations.append(m)
        susceptibilities.append(chi)
    
    magnetizations = np.array(magnetizations)
    susceptibilities = np.array(susceptibilities)
    
    # Fit de la susceptibilité: χ(L) ∝ L^(γ/ν)
    chi_params, chi_cov, gamma_over_nu, gamma_over_nu_err = fit_power_law(
        Ls, susceptibilities, "Susceptibilité χ(L) ∝ L^(γ/ν)"
    )
    
    # Fit de l'aimantation: M(L) ∝ L^(-β/ν)
    m_params, m_cov, minus_beta_over_nu, minus_beta_over_nu_err = fit_power_law(
        Ls, magnetizations, "Aimantation M(L) ∝ L^(-β/ν)"
    )
    
    beta_over_nu = -minus_beta_over_nu if minus_beta_over_nu is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Susceptibilité χ(L) en échelle linéaire
    ax = axes[0, 0]
    ax.plot(Ls, susceptibilities, 'o', color='darkblue', markersize=8, label='Données')
    if chi_params is not None:
        L_fit = np.linspace(Ls[0], Ls[-1], 100)
        chi_fit = power_law(L_fit, *chi_params)
        ax.plot(L_fit, chi_fit, 'r--', linewidth=2, 
                label=f'Fit: ${chi_params[0]:.2f} \\times L^{{{gamma_over_nu:.3f}}}$')
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
                  label=f'$\\gamma/\\nu = {gamma_over_nu:.3f} \\pm {gamma_over_nu_err:.3f}$')
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
                label=f'Fit: ${m_params[0]:.3f} \\times L^{{{minus_beta_over_nu:.3f}}}$')
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
                  label=f'$-\\beta/\\nu = {minus_beta_over_nu:.3f} \\pm {minus_beta_over_nu_err:.3f}$')
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