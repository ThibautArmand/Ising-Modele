# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from effet_champ_magnetique_exterieur.scaled_susceptibility import simulate_scaled_susceptibility_at_x

from utils.utils import (
    theoretical_magnetization,
)

@pytest.mark.slow
def test_first_order_peak_height_at_H0():
    """
    À T < Tc et H=0, Binder-Landau prédit : chi_L(0) / L^2 ≈ M_sp^2 / T
    """
    T = 2.1
    J = 1.0
    d = 2
    Ls = np.array([4, 6, 8])

    Msp = float(theoretical_magnetization(np.array([T]), J=J)[0])
    expected = Msp**2 / T

    values = np.array([
        simulate_scaled_susceptibility_at_x(
            L, T, J, x=0.0, d=d, 
            n_runs=10, 
            n_equilibration=1000, 
            n_measurements=500, 
            decorrelation_sweeps=2
        )
        for L in Ls
    ])

    measured = np.mean(values)

    tolerance = 0.10
    assert abs(measured - expected) / expected < tolerance, (
        f"chi_L(0)/L^2 = {measured:.3f}, attendu ≈ {expected:.3f}. "
        f"Valeurs par L : {values}"
    )

@pytest.mark.slow
def test_first_order_scaling_matches_binder_curve():
    """
    Compare chi_L/L^2 à la courbe analytique simplifiée de Binder-Landau : M_sp^2/T / cosh^2(x M_sp/T) pour une petite taille et quelques valeurs de x.
    """
    T = 2.1
    J = 1.0
    d = 2
    L = 6

    Msp = float(theoretical_magnetization(np.array([T]), J=J)[0])
    x_values = np.array([0.0, 1.0, 2.0])
    measured = np.array([
        simulate_scaled_susceptibility_at_x(
            L, T, J, x=x, d=d, 
            n_runs=10, 
            n_equilibration=1000, 
            n_measurements=500, 
            decorrelation_sweeps=2
        )
        for x in x_values
    ])

    expected = (Msp**2 / T) / np.cosh(x_values * Msp / T)**2
    rel_error = np.mean(np.abs(measured - expected) / expected)
    tolerance = 0.30
    assert rel_error < tolerance, (
        f"Écart trop grand à la courbe Binder-Landau. "
        f"mesuré={measured}, attendu={expected}, erreur relative moyenne={rel_error:.2f}"
    )