# -*- coding: utf-8 -*-
"""
Shared utility functions for the Ising model simulation.
"""
import numpy as np
from numba import njit

@njit
def remplissage_aleatoire_reseau(L):
    """
    Parameters
    ----------
    L : int
        Taille du réseau

    Returns
    -------
    tableau : [L,L]
    @Borrows
    -------
    - When `x=0`: `2*0-1 = -1`
    - When `x=1`: `2*1-1 = 1`
    
    """
    x = np.random.randint(0, 2, size=(L, L)) # \in {0, 1}
    return 2 * x - 1

@njit
def calculate_H(Reseau, i, j, J, h):
    """
    Parameters
    ----------
    Reseau : [L,L]
        Réseau de spins
    J : float, optional
        The default is 1.0.
    h : float, optional
        The default is 0.0.

    Returns
    -------
    H: float
        Énergie
    """
    # H = -J \sum S_k S_ij - h \sum S_ij=(-J\Sum S_k-h)*S_ij
    L,C=np.shape(Reseau) #L:nb lignes ; C:nb colonnes
    H = 0.0
    H=(-J * (
        Reseau[(i-1)%L,j] + 
        Reseau[(i+1)%L,j] + 
        Reseau[i,(j-1)%C] +
        Reseau[i,(j+1)%C]
        ) - h) * Reseau[i,j]
    return H
@njit
def calculate_dE(Reseau, i, j, J, h):
    """
    Parameters
    ----------
    spins : [L,L]
        Réseau
    i : int , indice ligne spin
    j : int , indice colonne spin
    J : float, optional
        The default is 1.0.
    h : float, optional
        The default is 0.0.

    Returns
    -------
    dE: float
    """    
    dE = -2*calculate_H(Reseau, i, j, J, h)  
    return dE

@njit
def MonteCarlo(n,N,Reseau,T,J,h):
    """
    Parameters
    ----------
    n : int
        Nombre de pas (entier naturel).
    N : int
        Taille du reseau (entier naturel)
    Reseau : [N,N]
        Tableau 2D de taille N*N
    T : float
        Température
    J : float
        constante d'échange
    h : float
        champ magnétique extérieur
    Returns
    -------
    Reseau : [N,N]

    """

    for k in range (n):
        i_s=np.random.randint(0,N)
        j_s=np.random.randint(0,N)
        dE = calculate_dE(Reseau, i_s, j_s, J, h)
        if dE <= 0 :
            Reseau[i_s, j_s] *= -1
        else :
            if np.random.rand() < np.exp(-dE / T):
                Reseau[i_s, j_s] *= -1
    return Reseau

@njit
def theoretical_magnetization(T, Tc=2.269, J=1.0):
    """
    Parameters
    ----------
    T : array
        Température
    Tc : float
        Température critique
    J : float
        Constante d'échange
    
    Returns
    -------
    m : array
    """
    m = np.zeros_like(T)
    mask = T < Tc
    m[mask] = (1 - np.sinh(2*J / T[mask])**(-4))**(1/8)
    return m

@njit
def calculate_magnetization(Reseau):
    """
    Parameters
    ----------
    Reseau : array [L,L]
        Réseau de spins
    
    Returns
    -------
    M : float
        Aimantation totale
    """
    return np.sum(Reseau)

@njit
def calculate_total_energy(Reseau, J=1.0, h=0.0):
    """
    Parameters
    ----------
    Reseau : array [L,L]
        Réseau de spins
    J : float
        Constante d'échange
    h : float
        Champ magnétique appliqué
    
    Returns
    -------
    E : float
        Énergie totale
    """
    L = Reseau.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            E += calculate_H(Reseau, i, j, J, h)
    return E

@njit
def simulate_single_temperature(L, T, n_equilibration=1000, n_measurements=1000, J=1.0, h=0.0, decorrelation_sweeps=1):
    """
    Parameters
    ----------
    L : int
        Taille du réseau
    T : float
        Température
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
    tuple : (e, m, C, chi)
        e : float - Énergie par spin
        m : float - Aimantation par spin
        C : float - Chaleur spécifique
        chi : float - Susceptibilité magnétique
    """
    N = L**2
    Reseau = remplissage_aleatoire_reseau(L)
    
    # Équilibration
    for _ in range(n_equilibration):
        Reseau = MonteCarlo(N, L, Reseau, T, J, h)
    
    energies = np.empty(n_measurements, dtype=np.float64)
    magnetizations_signed = np.empty(n_measurements, dtype=np.float64)
    magnetizations_abs = np.empty(n_measurements, dtype=np.float64)
    
    for i in range(n_measurements):
        for _ in range(decorrelation_sweeps):
            Reseau = MonteCarlo(N, L, Reseau, T, J, h)
        energies[i] = calculate_total_energy(Reseau, J, h)
        M = calculate_magnetization(Reseau)
        magnetizations_signed[i] = M
        magnetizations_abs[i] = np.abs(M)
    
    # Calcul des moyennes et fluctuations
    E_mean = np.mean(energies)
    E2_mean = np.mean(energies**2)
    M_mean_signed = np.mean(magnetizations_signed)
    M2_mean_signed = np.mean(magnetizations_signed**2)
    M_mean_abs = np.mean(magnetizations_abs)
    
    # Énergie par spin
    e = E_mean / N
    
    # Aimantation par spin
    m = M_mean_abs / N

    # Susceptibilité magnétique χ(T)
    # chi = (N / T) * (M2_mean_signed - M_mean_signed**2) 
    chi = (M2_mean_signed - M_mean_signed**2) / (N * T)

    # Chaleur spécifique  C(T)
    C = (1 / (T**2)) * (E2_mean - E_mean**2)
    
    return (e, m, C, chi)

def format_time(seconds):
    """
    Parameters
    ----------
    seconds : float
        Temps
    
    Returns
    -------
    str
    """
    if seconds < 1:
        return f"{seconds*1000:.2f} millisecondes"
    elif seconds < 60:
        return f"{seconds:.2f} secondes"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} et {secs:.2f} secondes"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} heure{'s' if hours > 1 else ''}, {minutes} minute{'s' if minutes > 1 else ''} et {secs:.2f} secondes"