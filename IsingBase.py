# -*- coding: utf-8 -*-
"""
Modèle d'Ising,transition de phase,méthode de Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt

def remplissage_aléatoire_reseau(N):
    #N:taille du réseau
    #retourne tableau taille [N,N]
    tableau=np.empty((N,N),dtype='int8')
    for i in range(len(tableau)):
        for j in range(len(tableau[i])):
            a=np.random.randint(0,2)
            if a==0:
                tableau[i][j]=-1
            elif a==1:
                tableau[i][j]=1
    return tableau

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

Reseau=remplissage_aléatoire_reseau(N)
plt.imshow(Reseau)
plt.imsave('reseau initial.png', Reseau, cmap='gray')
plt.show()

