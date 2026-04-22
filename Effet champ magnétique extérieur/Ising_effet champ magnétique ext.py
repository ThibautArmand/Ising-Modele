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

L=32 #taille du réseau : L*L
n = L**2 #definit un pas de MonteCarlo
J=1 #constante d'échange

#champ magnétique extérieur
amplitude_h=3
h=np.arange(-amplitude_h,amplitude_h,2*amplitude_h/100)
h=np.append(h,-h)
h=np.append(h,h)

T_s = [0.5, 1.5, 2.5, 3.5, 5.0]#température variable
m=[] # aimantations pour chaques températures

for i, T in enumerate(T_s):
    Reseau = remplissage_aleatoire_reseau(L)
    m_T=np.zeros(1)
    m_T[0]=np.mean(Reseau)
    
    for k in range(len(h)):
        Reseau = MonteCarlo(n, L, Reseau, T,J,h[k])
        m_T=np.append(m_T,np.mean(Reseau))
    m.append(m_T)

plt.plot(range(len(m[0])),m[0],'k-',linewidth=0.65,label=f'T={T_s[0]}')
plt.plot(range(len(m[1])),m[1],'r-',linewidth=0.65,label=f'T={T_s[1]}')
plt.plot(range(len(m[2])),m[2],'y-',linewidth=0.65,label=f'T={T_s[2]}')
plt.plot(range(len(m[3])),m[3],'b-',linewidth=0.65,label=f'T={T_s[3]}')
plt.plot(range(len(m[4])),m[4],'g-',linewidth=0.65,label=f'T={T_s[4]}')
#plt.plot(range(len(m[5])),m[5],'-',color='orange',linewidth=0.65,label=f'T={T_s[5]}')
plt.xticks(np.arange(0,len(m[0]),50))
plt.xlabel("nb pas",loc='right')
plt.ylabel(r'$ m $',loc='top')
plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
plt.title(f'Aimantation d\'un réseau de taille L={L} dans un champ magnétique d\'amplitude h={amplitude_h}')
plt.savefig(f'Evolution aimentation_L={L}_h={amplitude_h}.pdf', format="pdf",bbox_inches='tight')
plt.show()

for i, T in enumerate(T_s):
    plt.plot(h[100:200],m[i][101:201],'k-',linewidth=0.65,label=f'h de {h[100]} à {h[200]}')
    plt.plot(h[200:300],m[i][201:301],'r-',linewidth=0.65,label=f'h de {h[200]} à {h[300]}')
    plt.xlim(-amplitude_h, amplitude_h)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("h",loc='right')
    plt.ylabel(r'$ m $',loc='top')
    plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
    plt.title(f'Hystérésis de l\'aimantation - réseau de taille L={L} - champ magnétique d\'amplitude h={amplitude_h} à T={T}')
    plt.savefig(f'Hystérésis aimentation_L={L}_h={amplitude_h}_T={T}.pdf', format="pdf",bbox_inches='tight')
    plt.show()


temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")