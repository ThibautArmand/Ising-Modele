# -*- coding: utf-8 -*-
"""
Test suite pour vérifier le calcul du Cumulant de Binder et la détermination de la température critique T_c
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from Binder_Cumulant.binder_cumulant import (
    binder_cumulant,
    estimate_T_c,
    simulate_binder_cumulant
)
from utils.utils import simulate_M2_M4


def test_binder_cumulant_calculation():
    """
    Test que le calcul du cumulant de Binder donne des valeurs dans l'intervalle attendu
    
    Le cumulant de Binder U_L(T) = 1 - <M^4> / (3 <M^2>^2) devrait être:
    - U_L → 2/3 ≈ 0.667 pour T → 0 (phase ferromagnétique)
    - U_L → 0 pour T → ∞ (phase paramagnétique)
    """
    L = 16
    T_low = 1.0   # Basse température
    T_high = 4.0  # Haute température
    
    # Simulation à basse température
    M2_low, M4_low = simulate_M2_M4(L, T_low, n_equilibration=5000, n_measurements=2000)
    U_low = binder_cumulant(M2_low, M4_low)
    
    # Simulation à haute température
    M2_high, M4_high = simulate_M2_M4(L, T_high, n_equilibration=5000, n_measurements=2000)
    U_high = binder_cumulant(M2_high, M4_high)
    
    assert 0.0 <= U_low <= 1.0, f"U_L(T_low) = {U_low:.3f} n'est pas dans [0, 1]"
    assert -0.5 <= U_high <= 0.5, f"U_L(T_high) = {U_high:.3f} est hors de la plage attendue [-0.5, 0.5]"
    
    assert U_low > 0.4, f"U_L(T_low) = {U_low:.3f} devrait être > 0.4"
    assert abs(U_high) < 0.4, f"U_L(T_high) = {U_high:.3f} devrait être proche de 0"


def test_binder_cumulant_intersection_at_Tc():
    """
    Test que les courbes de Binder cumulant pour différentes tailles 
    s'intersectent près de la température critique théorique T_c = 2.269
    
    À T_c, les courbes de U_L(T) pour différentes tailles L devraient se croiser
    car le cumulant de Binder devient indépendant de la taille du système.
    """
    # Paramètres
    Tc_theoretical = 2.269
    Ls = [8, 16, 24]
    
    T_min = 2.0
    T_max = 2.5
    steps_T = 30
    T_s = np.linspace(T_min, T_max, steps_T)
    n_equilibration = 10000
    n_measurements = 5000
    
    binder_results = simulate_binder_cumulant(Ls, T_s, n_equilibration, n_measurements)
    T_c_estimate = estimate_T_c(T_s, Ls, binder_results)
    
    tolerance = 0.2  # ±20% de tolérance
    assert abs(T_c_estimate - Tc_theoretical) / Tc_theoretical < tolerance, \
        f"T_c estimée = {T_c_estimate:.3f} trop éloignée de T_c théorique = {Tc_theoretical:.3f} (écart: {abs(T_c_estimate - Tc_theoretical)/Tc_theoretical*100:.1f}%)"


def test_binder_cumulant_size_independence_at_Tc():
    """
    Test que le cumulant de Binder à T_c est approximativement indépendant de la taille
    
    À la température critique, U_L(T_c) devrait avoir des valeurs similaires
    pour différentes tailles L (avec des fluctuations statistiques).
    """
    # Paramètres
    Tc = 2.269
    Ls = [8, 16, 32]
    n_equilibration = 5000
    n_measurements = 3000
    U_values = [] # Calcul du cumulant de Binder à T_c pour différentes tailles
    
    for L in Ls:
        M2_mean, M4_mean = simulate_M2_M4(
            L, Tc, n_equilibration, n_measurements, decorrelation_sweeps=5
        )
        U_L = binder_cumulant(M2_mean, M4_mean)
        U_values.append(U_L)
    
    U_values = np.array(U_values)
    variance = np.var(U_values)
    mean_U = np.mean(U_values)
    cv = np.sqrt(variance) / abs(mean_U) if mean_U != 0 else np.inf
    
    assert cv < 0.2, \
        f"Coefficient de variation {cv:.3f} trop élevé. U_L devrait être indépendant de L à T_c. Valeurs: {U_values}"

