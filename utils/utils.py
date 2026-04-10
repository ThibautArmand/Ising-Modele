# -*- coding: utf-8 -*-
"""
Shared utility functions for the Ising model simulation.
"""
import numpy as np
from numba import njit

#@njit
def remplissage_aleatoire_reseau(L):
    """
    Parameters
    ----------
    L : int
        Taille du réseau

    Returns
    -------
    tableau : [L,L]

    """
    return np.random.choice([-1, 1], size=(L, L))
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
    T : float or array
        Température
    Tc : float
        Température critique
    J : float
        Constante d'échange
    
    Returns
    -------
    m : float or array
    """
    T = np.array(T)
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