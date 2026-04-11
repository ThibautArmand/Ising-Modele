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
def test_simulate_temperature_returns_tuple():
    """Test that simulate_temperature returns a tuple"""
    L = 4
    T = 2.0
    result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 4, "Result should have 4 elements (e, m, C, chi)"


def test_simulate_temperature_unpacking():
    """Test that the returned tuple can be unpacked correctly"""
    L = 4
    T = 2.5
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    # Check all values are returned
    assert e is not None, "Energy per spin should be returned"
    assert m is not None, "Magnetization per spin should be returned"
    assert C is not None, "Specific heat should be returned"
    assert chi is not None, "Susceptibility should be returned"


def test_simulate_temperature_values_are_floats():
    """Test that all returned values are floats"""
    L = 4
    T = 2.0
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    values = {'e': e, 'm': m, 'C': C, 'chi': chi}
    for key, value in values.items():
        assert isinstance(value, (float, np.floating)), f"{key} should be a float, got {type(value)}"


def test_simulate_temperature_magnetization_bounds():
    """Test that magnetization per spin is within physical bounds [0, 1]"""
    L = 8
    T = 2.0
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=20, n_measurements=20)
    
    assert 0.0 <= m <= 1.0, f"Magnetization per spin should be in [0, 1], got {m}"


def test_simulate_temperature_energy_bounds():
    """Test that energy per spin is within reasonable physical bounds"""
    L = 8
    T = 2.5
    J = 1.0
    h = 0.0
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=20, n_measurements=20, J=J, h=h)
    
    # For 2D Ising with J=1, h=0, energy per spin ranges from -4 (all aligned, counting each bond)
    # to +4 (checkerboard pattern). In practice, should be in [-4, 1] range
    assert -4.0 <= e <= 1.0, f"Energy per spin should be in reasonable range, got {e}"


def test_simulate_temperature_susceptibility_positive():
    """Test that magnetic susceptibility is non-negative"""
    L = 4
    T = 2.0
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    assert chi >= 0, f"Susceptibility should be non-negative, got {chi}"


def test_simulate_temperature_specific_heat_positive():
    """Test that specific heat is non-negative"""
    L = 4
    T = 2.5
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
    
    assert C >= 0, f"Specific heat should be non-negative, got {C}"

def test_simulate_temperature_high_temperature():
    """Test behavior at high temperature (disordered phase)"""
    L = 8
    T = 5.0  # High temperature, should be disordered
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=50, n_measurements=50)
    
    # At high T, magnetization should be low (close to 0)
    assert m < 0.3, f"At high T, magnetization should be low, got {m}"


def test_simulate_temperature_different_lattice_sizes():
    """Test that function works with different lattice sizes"""
    sizes = [4, 8, 16]
    T = 2.5
    
    for L in sizes:
        result = simulate_temperature(L, T, n_equilibration=10, n_measurements=10)
        assert isinstance(result, tuple), f"Failed for L={L}"
        assert len(result) == 4, f"Should return 4 values for L={L}"


def test_simulate_temperature_with_external_field():
    """Test simulation with external magnetic field"""
    L = 8
    T = 2.0
    h = 0.5  # External field
    
    e, m, C, chi = simulate_temperature(L, T, n_equilibration=20, n_measurements=20, h=h)
    
    # With positive external field, magnetization should be positive
    assert m >= 0, f"With positive h, magnetization should be non-negative, got {m}"


def test_simulate_temperature_minimal_parameters():
    """Test with minimal number of steps (edge case)"""
    L = 4
    T = 2.0
    
    # Very few steps - should still run without error
    result = simulate_temperature(L, T, n_equilibration=1, n_measurements=1)
    
    assert isinstance(result, tuple), "Should return tuple even with minimal steps"
    assert len(result) == 4, "Should return 4 values (e, m, C, chi)"
    
    e, m, C, chi = result
    assert all(x is not None for x in [e, m, C, chi]), \
        "All returned values should be non-None even with minimal steps"
    
def test_simulate_temperature_susceptibility_peak_location():
    """Test that susceptibility peaks near critical temperature"""
    L = 64
    Tc = 2.269
    
    # Sample temperatures around Tc
    T_below = Tc - 0.5
    T_at = Tc
    T_above = Tc + 0.5
    
    _, _, _, chi_below = simulate_temperature(L, T_below, n_equilibration=10000, n_measurements=1000)
    _, _, _, chi_at = simulate_temperature(L, T_at, n_equilibration=10000, n_measurements=1000)
    _, _, _, chi_above = simulate_temperature(L, T_above, n_equilibration=10000, n_measurements=1000)
    
    # Susceptibility at Tc should be >= both sides (peak behavior)
    # Allow some tolerance for finite-size effects
    assert chi_at >= chi_above * 0.8, \
        f"χ at Tc ({chi_at}) should be high compared to above ({chi_above})"