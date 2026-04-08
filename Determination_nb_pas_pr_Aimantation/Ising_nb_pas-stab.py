# -*- coding: utf-8 -*-
"""
Modèle d'Ising,transition de phase,méthode de Monte Carlo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from utils.utils import (
    remplissage_aleatoire_reseau,
    calculate_H,
    calculate_dE,
    MonteCarlo
)

N=64 #taille du réseau : N*N
n = N**2
T_s = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]#température variable
m=[] # aimantations pour chaques températures

fig, axes = plt.subplots(1,6,figsize=(10, 10))
axes = axes.flatten()

for i, T in enumerate(T_s):
    ax = axes[i]
    Reseau = remplissage_aleatoire_reseau(N)
    
    # Evolution de l'aimantation
    m_T=np.zeros(1)
    m_T[0]=np.abs(np.mean(Reseau))

    for k in range(5):
        Reseau = MonteCarlo(n, N, Reseau, T)
        m_T=np.append(m_T,np.abs(np.mean(Reseau)))

    # Continuer tant que l'aimantation varie plus du 5 % 
    ## m_T[-1] is closer to m_T[-2] OR
    ## m_T[-1] is closer to m_T[-3]
    while ( abs(m_T[-1] - m_T[-2]) > 0.05 * abs(m_T[-2]) or 
            abs(m_T[-1] - m_T[-3]) > 0.05 * abs(m_T[-3])):
        Reseau = MonteCarlo(n, N, Reseau, T)
        m_T=np.append(m_T,np.abs(np.mean(Reseau)))
    
    m.append(m_T)
    ax.imshow(Reseau)
    ax.set_title(f'T = {T}')
    print(m[i])

plt.savefig('ising_aimantation stabilisée.pdf', bbox_inches='tight')
plt.show()

plt.plot(range(len(m[0])),m[0],'k-',linewidth=0.65,label=f'T={T_s[0]}')
plt.plot(range(len(m[1])),m[1],'r-',linewidth=0.65,label=f'T={T_s[1]}')
plt.plot(range(len(m[2])),m[2],'y-',linewidth=0.65,label=f'T={T_s[2]}')
plt.plot(range(len(m[3])),m[3],'b-',linewidth=0.65,label=f'T={T_s[3]}')
plt.plot(range(len(m[4])),m[4],'g-',linewidth=0.65,label=f'T={T_s[4]}')
plt.plot(range(len(m[5])),m[5],'-',color='orange',linewidth=0.65,label=f'T={T_s[5]}')
plt.xlabel("nb pas",loc='right')
plt.ylabel(r'$ | m | $',loc='top')
plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
plt.title("Evolution de l'Aimantation suivant un pas de MonteCorlo pour différentes tepératures")
plt.savefig("Evolution de l'aimentation.pdf", format="pdf",bbox_inches='tight')
plt.show()