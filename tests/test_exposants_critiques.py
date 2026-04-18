# -*- coding: utf-8 -*-
"""
Test suite pour vérifier que les exposants critiques numériques 
sont proches des valeurs théoriques du modèle d'Ising 2D
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from scipy.optimize import curve_fit

from utils.utils import simulate_single_temperature


def power_law(x, a, b):
    """Loi de puissance pour le fitting"""
    return a * x**b


def test_exposant_critique_gamma_nu():
    """
    Test que l'exposant critique γ/ν est proche de la valeur théorique 7/4 = 1.75
    
    Pour le modèle d'Ising 2D, à T = Tc:
    χ(L) ∝ L^(γ/ν) avec γ/ν = 1.75 (théorique)
    """
    # Paramètres
    Tc = 2.269
    Ls = np.array([8, 16, 32, 48])  # Tailles réduites pour test rapide
    gamma_nu_theorique = 7/4  # = 1.75
    
    susceptibilities = []
    
    # Simulations
    for L in Ls:
        n_equilibration = 5000 if L <= 32 else 10000
        n_measurements = 2000 if L <= 32 else 5000
        _, _, _, chi = simulate_single_temperature(L, Tc, n_equilibration, n_measurements)
        susceptibilities.append(chi)
    
    susceptibilities = np.array(susceptibilities)
    
    # Fit: χ(L) ∝ L^(γ/ν)
    params, _ = curve_fit(power_law, Ls, susceptibilities, p0=[1.0, 1.0])
    gamma_nu_estimation = params[1]
    
    # Vérification avec tolérance de 15% (les simulations MC ont des fluctuations)
    tolerance = 0.15
    assert abs(gamma_nu_estimation - gamma_nu_theorique) / gamma_nu_theorique < tolerance, \
        f"γ/ν = {gamma_nu_estimation:.3f} trop éloigné de la valeur théorique {gamma_nu_theorique:.3f}"


def test_exposant_critique_beta_nu():
    """
    Test que l'exposant critique -β/ν est proche de la valeur théorique -1/8 = -0.125
    
    Pour le modèle d'Ising 2D, à T = Tc:
    M(L) ∝ L^(-β/ν) avec -β/ν = -0.125 (théorique)
    """
    # Paramètres
    Tc = 2.269
    Ls = np.array([8, 16, 32, 48])  # Tailles réduites pour test rapide
    beta_nu_theorique = -1/8  # = -0.125
    
    magnetizations = []
    
    # Simulations
    for L in Ls:
        n_equilibration = 5000 if L <= 32 else 10000
        n_measurements = 2000 if L <= 32 else 5000
        _, m, _, _ = simulate_single_temperature(L, Tc, n_equilibration, n_measurements)
        magnetizations.append(m)
    
    magnetizations = np.array(magnetizations)
    
    # Fit: M(L) ∝ L^(-β/ν)
    params, _ = curve_fit(power_law, Ls, magnetizations, p0=[1.0, -0.1])
    beta_nu_estimation = params[1]
    
    # Vérification avec tolérance de 20% (β/ν est plus difficile à estimer précisément)
    tolerance = 0.20
    assert abs(beta_nu_estimation - beta_nu_theorique) / abs(beta_nu_theorique) < tolerance, \
        f"-β/ν = {beta_nu_estimation:.3f} trop éloigné de la valeur théorique {beta_nu_theorique:.3f}"
