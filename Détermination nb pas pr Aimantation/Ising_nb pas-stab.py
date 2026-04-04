# -*- coding: utf-8 -*-
"""
Modèle d'Ising,transition de phase,méthode de Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt

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
    H=(-J*(Reseau[(i-1)%L,j]+Reseau[(i+1)%L,j]+Reseau[i,(j-1)%C]+Reseau[i,(j+1)%C])-h)*Reseau[i,j]
    return H

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

def MonteCarlo(n,N,Reseau,T):
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
    Returns
    -------
    Reseau : [N,N]

    """
    J=1
    h=0

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

N=32 #taille du réseau : N*N
n = N**2  # how many steps do we need to equilibrate the system ?
T_s = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]

fig, axes = plt.subplots(1,6,figsize=(10, 10))
axes = axes.flatten()

for i, T in enumerate(T_s):
    ax = axes[i]
    Reseau = remplissage_aleatoire_reseau(N)
    for k in range(100):
        Reseau = MonteCarlo(n, N, Reseau, T)
    
    ax.imshow(Reseau)
    ax.set_title(f'T = {T}')
plt.savefig('ising_aimantation stabilisée.pdf', bbox_inches='tight')
plt.show()

