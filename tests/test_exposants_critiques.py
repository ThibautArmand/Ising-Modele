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

from Transition_Phase_Tailles.exposants_critiques import simulate_finite_size_scaling


def test_exposant_critique_gamma_nu():
    """
    Test que l'exposant critique γ/ν est proche de la valeur théorique 7/4 = 1.75
    
    Pour le modèle d'Ising 2D, à T = Tc:
    χ(L) ∝ L^(γ/ν) avec γ/ν = 1.75 (théorique)
    """
    # Paramètres
    Tc = 2.269
    Ls = np.array([8, 16, 32, 48])
    gamma_nu_theorique = 7/4  # = 1.75
    
    (gamma_nu_estimation, gamma_error, _, _), _, _ = simulate_finite_size_scaling(Ls, Tc)
    
    # Vérification avec tolérance de 20%
    tolerance = 0.20
    assert abs(gamma_nu_estimation - gamma_nu_theorique) / gamma_nu_theorique < tolerance, \
        f"γ/ν = {gamma_nu_estimation:.3f} ± {gamma_error:.3f} trop éloigné de la valeur théorique {gamma_nu_theorique:.3f}"


def test_exposant_critique_beta_nu():
    """
    Test que l'exposant critique -β/ν est proche de la valeur théorique -1/8 = -0.125
    
    Pour le modèle d'Ising 2D, à T = Tc:
    M(L) ∝ L^(-β/ν) avec -β/ν = -0.125 (théorique)
    """
    # Paramètres
    Tc = 2.269
    Ls = np.array([8, 16, 32, 48])
    beta_nu_theorique = -1/8  # = -0.125
    
    (_, _, beta_nu_estimation, beta_error), _, _ = simulate_finite_size_scaling(Ls, Tc)
    
    # Vérification avec tolérance de 25%
    tolerance = 0.25
    assert abs(beta_nu_estimation - beta_nu_theorique) / abs(beta_nu_theorique) < tolerance, \
        f"-β/ν = {beta_nu_estimation:.3f} ± {beta_error:.3f} trop éloigné de la valeur théorique {beta_nu_theorique:.3f}"
