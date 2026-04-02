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
    L : int8
        Taille du réseau

    Returns
    -------
    tableau : [L,L]

    """
    return np.random.choice([-1, 1], size=(L, L))

def calculate_H(spins, J = 1.0, h = 0.0):
    """
    Parameters
    ----------
    spins : [L,L]
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
    L = spins.shape[0]

    # H - J \sum S_i S_j - h \sum S_i
    H = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            # Pairs 
            # right neighbor
            # to handle limits
            # i.e : if j = 0 : (0+1)%8 = 1
            s_r = spins[i, (j + 1) % L]
            # bottom neighbor
            s_b = spins[(i + 1) % L, j]
            # - J \sum S_i S_j
            # -J s s_{droite} - J s s_{bas}
            H += -J * s * (s_r + s_b)
    # remaining expression
    H += -h * np.sum(spins)
    return H

def calculate_dE(spins, i, j, J=1.0, h=0.0):
    """
    Parameters
    ----------
    spins : [L,L]
        Réseau
    i : int8
    j : int8
    J : float, optional
        The default is 1.0.
    h : float, optional
        The default is 0.0.

    Returns
    -------
    dE: float
    """    
    H_old = calculate_H(spins, J, h)
    
    spins_new = spins.copy()
    spins_new[i, j] *= -1
    
    H_new = calculate_H(spins_new, J, h)
   
    return H_new - H_old

def MonteCarlo(n,N,Reseau,T):
    """
    Parameters
    ----------
    n : int8
        Nombre de pas (entier naturel).
    N : int8
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

    i_s=np.random.randint(0,N, size=n)
    j_s=np.random.randint(0,N, size=n)
    r_s = np.random.rand(n)

    for k in range (n):
        i, j, r = i_s[k], j_s[k], r_s[k]
        dE = calculate_dE(Reseau, i, j, J, h)
        if dE <= 0 or r < np.exp(-dE / T):
            Reseau[i, j] *= -1 
    return Reseau

N=32 #taille du réseau : N*N
n = 10000  # how many steps do we need to equilibrate the system ?
T_s = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]

fig, axes = plt.subplots(1,6,figsize=(10, 10))
axes = axes.flatten()

for i, T in enumerate(T_s):
    ax = axes[i]
    Reseau = Reseau=remplissage_aleatoire_reseau(N)
    Reseau = MonteCarlo(n, N, Reseau, T)
    
    ax.imshow(Reseau)
    ax.set_title(f'T = {T}')
plt.savefig('ising_base.pdf', bbox_inches='tight')
plt.show()

