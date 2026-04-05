# -*- coding: utf-8 -*-
"""
Test suite for utils functions
"""
import numpy as np
from pathlib import Path


from utils.utils import (
    remplissage_aleatoire_reseau,
    calculate_H,
    calculate_dE,
    MonteCarlo
)

# Tests for remplissage_aleatoire_reseau
def test_remplissage_aleatoire_reseau_shape():
    """Test that the lattice has the correct shape"""
    L = 10
    reseau = remplissage_aleatoire_reseau(L)
    assert reseau.shape == (L, L), f"Expected shape ({L}, {L}), got {reseau.shape}"


def test_remplissage_aleatoire_reseau_values():
    """Test that all values are either -1 or 1"""
    L = 5
    reseau = remplissage_aleatoire_reseau(L)
    unique_values = np.unique(reseau)
    assert set(unique_values).issubset({-1, 1}), f"Expected only -1 and 1, got {unique_values}"


# Tests for calculate_H
def test_calculate_H_uniform_positive():
    """Test energy calculation for uniform positive spins"""
    # All spins = 1, should give negative energy (ferromagnetic coupling)
    # [[1, 1, 1],
    #  [1, 1, 1],
    #  [1, 1, 1]]
    reseau = np.ones((3, 3))
    J = 1.0
    h = 0.0
    H = calculate_H(reseau, 1, 1, J, h)
    # H = (-J * (4 neighbors all = 1) - h) * S_ij = (-1 * 4 - 0) * 1 = -4
    assert H == -4.0, f"Expected H = -4.0, got {H}"


def test_calculate_H_uniform_negative():
    """Test energy calculation for uniform negative spins"""
    # All spins = -1
    reseau = -np.ones((3, 3))
    # [[-1, -1, -1],
    #  [-1, -1, -1],
    #  [-1, -1, -1]]
    J = 1.0
    h = 0.0
    H = calculate_H(reseau, 1, 1, J, h)
    # H = (-J * (4 neighbors all = -1) - h) * S_ij = (-1 * (-4) - 0) * (-1) = -4
    assert H == -4.0, f"Expected H = -4.0, got {H}"

def test_calculate_H_mixed():
    """Test energy calculation for a mixed spin configuration"""
    reseau = np.array([[1, -1, 1],
                       [-1, 1, -1],
                       [1, -1, 1]])
    # Central spin = 1, neighbors = -1
    J = 1.0
    h = 0.0
    H = calculate_H(reseau, 1, 1, J, h)
    # H = (-J * (4 neighbors all = -1) - h) * S_ij = (-1 * (-4) - 0) * 1 = 4
    assert H == 4.0, f"Expected H = 4.0, got {H}"


# Tests for calculate_dE
def test_calculate_dE_uniform():
    """Test energy difference for flipping a spin in uniform field"""
    reseau = np.ones((3, 3))
    J = 1.0
    h = 0.0
    dE = calculate_dE(reseau, 1, 1, J, h)
    # dE = -2 * H = -2 * (-4) = 8
    assert dE == 8.0, f"Expected dE = 8.0, got {dE}"


def test_calculate_dE_checkerboard():
    """Test energy difference in a checkerboard pattern"""
    reseau = np.array([[1, -1, 1],
                       [-1, 1, -1],
                       [1, -1, 1]])
    J = 1.0
    h = 0.0
    dE = calculate_dE(reseau, 1, 1, J, h)
    # Central spin = 1, all 4 neighbors = -1
    # H = (-1 * (-4) - 0) * 1 = 4
    # dE = -2 * 4 = -8
    assert dE == -8.0, f"Expected dE = -8.0, got {dE}"


# Tests for MonteCarlo
def test_MonteCarlo_returns_same_shape():
    """Test that Monte Carlo preserves lattice shape"""
    N = 5
    n_steps = 10
    T = 2.0
    reseau = remplissage_aleatoire_reseau(N)
    reseau_final = MonteCarlo(n_steps, N, reseau.copy(), T)
    assert reseau_final.shape == (N, N), f"Shape changed after Monte Carlo"


def test_MonteCarlo_values_remain_valid():
    """Test that Monte Carlo only produces -1 or 1 values"""
    N = 5
    n_steps = 100
    T = 2.5
    reseau = remplissage_aleatoire_reseau(N)
    reseau_final = MonteCarlo(n_steps, N, reseau.copy(), T)
    unique_values = np.unique(reseau_final)
    assert set(unique_values).issubset({-1, 1}), f"Invalid spin values after Monte Carlo: {unique_values}"

