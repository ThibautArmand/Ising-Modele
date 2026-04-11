# -*- coding: utf-8 -*-
"""
Test suite for transition_phase functions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from Transition_Phase_Tailles.transition_phase import simulate_temperature

# Tests for simulate_temperature
def test_simulate_temperature_returns_dict():
    """Test that simulate_temperature returns a dictionary"""
    L = 4
    T = 2.0
    result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    assert isinstance(result, dict), "Result should be a dictionary"


def test_simulate_temperature_has_correct_keys():
    """Test that the returned dictionary has all required keys"""
    L = 4
    T = 2.5
    result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    required_keys = {'energie_per_spin', 'magnetization_per_spin', 'specific_heat', 'susceptibility'}
    assert set(result.keys()) == required_keys, f"Expected keys {required_keys}, got {set(result.keys())}"


def test_simulate_temperature_values_are_floats():
    """Test that all returned values are floats"""
    L = 4
    T = 2.0
    result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    for key, value in result.items():
        assert isinstance(value, (float, np.floating)), f"{key} should be a float, got {type(value)}"


def test_simulate_temperature_magnetization_bounds():
    """Test that magnetization per spin is within physical bounds [0, 1]"""
    L = 8
    T = 2.0
    result = simulate_temperature(L, T, n_equilibration=20, n_measurements=20)
    
    m = result['magnetization_per_spin']
    assert 0.0 <= m <= 1.0, f"Magnetization per spin should be in [0, 1], got {m}"


def test_simulate_temperature_energy_bounds():
    """Test that energy per spin is within reasonable physical bounds"""
    L = 8
    T = 2.5
    J = 1.0
    h = 0.0
    result = simulate_temperature(L, T, n_equilibration=20, n_measurements=20, J=J, h=h)
    
    e = result['energie_per_spin']
    # For 2D Ising with J=1, h=0, energy per spin ranges from -4 (all aligned, counting each bond)
    # to +4 (checkerboard pattern). In practice, should be in [-4, 1] range
    assert -4.0 <= e <= 1.0, f"Energy per spin should be in reasonable range, got {e}"


def test_simulate_temperature_susceptibility_positive():
    """Test that magnetic susceptibility is non-negative"""
    L = 4
    T = 2.0
    result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    chi = result['susceptibility']
    assert chi >= 0, f"Susceptibility should be non-negative, got {chi}"


def test_simulate_temperature_specific_heat_positive():
    """Test that specific heat is non-negative"""
    L = 4
    T = 2.5
    result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    C = result['specific_heat']
    assert C >= 0, f"Specific heat should be non-negative, got {C}"

def test_simulate_temperature_high_temperature():
    """Test behavior at high temperature (disordered phase)"""
    L = 8
    T = 5.0  # High temperature, should be disordered
    result = simulate_temperature(L, T, n_equilibration=50, n_measurements=50)
    
    m = result['magnetization_per_spin']
    # At high T, magnetization should be low (close to 0)
    assert m < 0.3, f"At high T, magnetization should be low, got {m}"


def test_simulate_temperature_different_lattice_sizes():
    """Test that function works with different lattice sizes"""
    sizes = [4, 8, 16]
    T = 2.5
    
    for L in sizes:
        result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
        assert isinstance(result, dict), f"Failed for L={L}"
        assert 'magnetization_per_spin' in result, f"Missing magnetization for L={L}"


def test_simulate_temperature_with_external_field():
    """Test simulation with external magnetic field"""
    L = 8
    T = 2.0
    h = 0.5  # External field
    
    result = simulate_temperature(L, T, n_equilibration=20, n_measurements=20, h=h)
    
    m = result['magnetization_per_spin']
    # With positive external field, magnetization should be positive
    assert m >= 0, f"With positive h, magnetization should be non-negative, got {m}"


def test_simulate_temperature_minimal_parameters():
    """Test with minimal number of steps (edge case)"""
    L = 4
    T = 2.0
    
    # Very few steps - should still run without error
    result = simulate_temperature(L, T, n_equilibration=1, n_measurements=1)
    
    assert isinstance(result, dict), "Should return dict even with minimal steps"
    assert all(key in result for key in ['energie_per_spin', 'magnetization_per_spin', 
                                           'specific_heat', 'susceptibility']), \
        "Should have all required keys even with minimal steps"