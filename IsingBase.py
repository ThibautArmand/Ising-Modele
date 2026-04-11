# -*- coding: utf-8 -*-
"""
Modèle d'Ising,transition de phase,méthode de Monte Carlo
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import (
    remplissage_aleatoire_reseau,
    calculate_H,
    calculate_dE,
    MonteCarlo,
    format_time
)
temps_init = time.time()

N=32 #taille du réseau : N*N
n = N**2  # how many steps do we need to equilibrate the system ?
J = 1.0
h = 0.0  # champ magnétique extérieur
T_s = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]

fig, axes = plt.subplots(1,6,figsize=(10, 10))
axes = axes.flatten()

for i, T in enumerate(T_s):
    ax = axes[i]
    Reseau = remplissage_aleatoire_reseau(N)
    for k in range(100):
        Reseau = MonteCarlo(n, N, Reseau, T, J, h)
    
    ax.imshow(Reseau)
    ax.set_title(f'T = {T}')
plt.savefig('ising_base.pdf', bbox_inches='tight')
plt.show()

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")

