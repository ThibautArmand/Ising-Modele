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

N=32 #taille du réseau : N*N
n = N**2 #definit un pas de MonteCarlo
J=1 #constante d'échange
h=0 #champ magnétique extérieur
T_s = [0.5, 2.0, 3.5, 5.0]#température variable
m=[] # aimantations pour chaques températures

for i, T in enumerate(T_s):
    Reseau = remplissage_aleatoire_reseau(N)
    m_T=np.zeros(1)
    m_T[0]=np.mean(Reseau)
    h=-2.5
    for k in range(50):
        Reseau = MonteCarlo(n, N, Reseau, T,J,h)
        m_T=np.append(m_T,np.mean(Reseau))
    h=2.5
    for k in range(50):
        Reseau = MonteCarlo(n, N, Reseau, T,J,h)
        m_T=np.append(m_T,np.mean(Reseau))
    h=-2.5
    for k in range(50):
        Reseau = MonteCarlo(n, N, Reseau, T,J,h)
        m_T=np.append(m_T,np.mean(Reseau))
    m.append(m_T)

plt.plot(range(len(m[0])),m[0],'k-',linewidth=0.65,label=f'T={T_s[0]}')
plt.plot(range(len(m[1])),m[1],'r-',linewidth=0.65,label=f'T={T_s[1]}')
plt.plot(range(len(m[2])),m[2],'y-',linewidth=0.65,label=f'T={T_s[2]}')
plt.plot(range(len(m[3])),m[3],'b-',linewidth=0.65,label=f'T={T_s[3]}')
#plt.plot(range(len(m[4])),m[4],'g-',linewidth=0.65,label=f'T={T_s[4]}')
#plt.plot(range(len(m[5])),m[5],'-',color='orange',linewidth=0.65,label=f'T={T_s[5]}')
plt.xticks(np.arange(0,150,25))
plt.xlabel("nb pas",loc='right')
plt.ylabel(r'$ | m | $',loc='top')
plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
plt.title(f'Aimantation en présence d\'un champ magnétique variant de h=-2.5 à +2.5,N={N},k=50')
plt.savefig(f'Evolution aimentation-champ magnétique h=-2.5@+2.5@-2.5_N={N}_k=50.pdf', format="pdf",bbox_inches='tight')
plt.show()