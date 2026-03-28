# -*- coding: utf-8 -*-
"""
Modèle d'Ising,transition de phase,méthode de Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt

def remplissage_aléatoire_reseau(L):
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
    L = spins.shape[0]
    s = spins[i, j]
    
    H_old = calculate_H(spins, J, h)
    
    spins_new = spins.copy()
    spins_new[i, j] *= -1
    
    H_new = calculate_H(spins_new, J, h)
   
    return H_new - H_old

def MonteCarlo(n,N,Reseau,T):
    #n:nombre de pas (entier naturel)
    #N:taille du reseau (entier naturel)
    #Reseau: tableau 2D de taille N*N
    #T: température (float)
    #retourne un tableau 2D
    J=1
    h=0
    for i in range (n):
        x=np.random.randint(0,N+1)
        y=np.random.randint(0,N+1)
        

N=32 #taille du réseau : N*N

# this is a comment

Reseau=remplissage_aléatoire_reseau(N)
plt.imshow(Reseau)
plt.imsave('reseau initial.png', Reseau, cmap='gray')
plt.show()

