# -*- coding: utf-8 -*-
"""
Modèle d'Ising,transition de phase,méthode de Monte Carlo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

Nvalues=np.array([4,8,16,32,64,128]) #valeurs de tailles du réseau : N*N
nvalues = Nvalues*Nvalues #definit un pas de Monte Carlo pour chaque taille
J=1 #constante d'échange
h=0 #champ magnétique extérieur
T_s = [0.5, 1.5, 2.5, 3.5, 5.0]#température variable

for j, N in enumerate(Nvalues):
    n=nvalues[j]
    
    fig, axes = plt.subplots(1,5,figsize=(10, 10))
    axes = axes.flatten()
    
    m=[] # aimantations pour chaques températures
    for i, T in enumerate(T_s):
        ax = axes[i]
        Reseau = remplissage_aleatoire_reseau(N)
        m_T=np.zeros(1)
        m_T[0]=np.abs(np.mean(Reseau))
        for k in range(5):
            Reseau = MonteCarlo(n, N, Reseau, T,J,h)
            m_T=np.append(m_T,np.abs(np.mean(Reseau)))
        while(1.05*m_T[-1]<m_T[-2] or 0.95*m_T[-1]>m_T[-2] or 1.05*m_T[-1]<m_T[-3] or 0.95*m_T[-1]>m_T[-3]):
            Reseau = MonteCarlo(n, N, Reseau, T,J,h)
            m_T=np.append(m_T,np.abs(np.mean(Reseau)))
    
        m.append(m_T)
        ax.imshow(Reseau)
        ax.set_title(f'T = {T}')

    plt.savefig(f'ising_aimantation stabilisée-critère 5% moyenne-N={N}.pdf', bbox_inches='tight')
    plt.show()

    plt.plot(range(len(m[0])),m[0],'k-',linewidth=0.65,label=f'T={T_s[0]}')
    plt.plot(range(len(m[1])),m[1],'r-',linewidth=0.65,label=f'T={T_s[1]}')
    plt.plot(range(len(m[2])),m[2],'y-',linewidth=0.65,label=f'T={T_s[2]}')
    plt.plot(range(len(m[3])),m[3],'b-',linewidth=0.65,label=f'T={T_s[3]}')
    plt.plot(range(len(m[4])),m[4],'k--',linewidth=0.65,label=f'T={T_s[4]}')
    #plt.plot(range(len(m[5])),m[5],'r--',color='orange',linewidth=0.65,label=f'T={T_s[5]}')
    plt.xlabel("nb pas",loc='right')
    plt.ylabel(r'$ | m | $',loc='top')
    plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
    plt.title(f'Evolution de l\'aimantation sur critère +/-5% de la moyenne,N={N}')
    plt.savefig(f'Evolution de l\'aimentation-critère 5% moyenne_N={N}.pdf', format="pdf",bbox_inches='tight')
    plt.show()

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")