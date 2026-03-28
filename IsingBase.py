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

