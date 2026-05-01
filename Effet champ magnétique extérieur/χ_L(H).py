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

Lvalues=np.array([4,6,8,32]) #taille du réseau : L*L
n = Lvalues*Lvalues #definit un pas de MonteCarlo
J=1 #constante d'échange
d = 2

#champ magnétique extérieur
amplitude_h=5
taille_h=200
h_up=np.arange(-amplitude_h,amplitude_h,2*amplitude_h/taille_h)
h_down=np.arange(amplitude_h,-amplitude_h,-2*amplitude_h/taille_h)
h=np.concatenate([h_up, h_down])

T_s = [0.5,1.5,2.0,5]#température variable

for i, T in enumerate(T_s):
    beta = 1/T
    #m=[] # aimantations pour chaque taille L
    for j, L in enumerate(Lvalues):
        Reseau = remplissage_aleatoire_reseau(L)
        m_L=np.zeros(len(h))

        n_sweeps = 200
        n_steps = n_sweeps * L * L
        
        for k in range(len(h)):
            Reseau = MonteCarlo(n_steps, L, Reseau, T,J,h[k])
            m_L[k]=np.mean(Reseau)
        
        m_up = m_L[:taille_h]
        m_down = m_L[taille_h:]

        idx_zero = np.argmin(np.abs(h_up))
        M_L = np.abs(m_up[idx_zero])
        chi_num = np.gradient(m_up, h_up)

        chi_D = 0.0
        chi_th = chi_D + (beta * M_L**2 * L**d) / (np.cosh(beta * h_up * M_L * L**d))**2

        fig , (ax1,ax2) = plt.subplots(1,2,figsize=(10, 10))
        ax1.plot(h_up,m_up,'k-',linewidth=0.65,label=f'h croissant: {h_up[0]:.2f} à {h_up[-1]:.2f}')
        ax1.plot(h_down,m_down,'r-',linewidth=0.65,label=f'h décroissant: {h_up[0]:.2f} à {h_up[-1]:.2f}')
        ax1.set_xlim(-amplitude_h, amplitude_h)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlabel("h",loc='right')
        ax1.set_ylabel(r'$ m $',loc='top')
        #ax1.set_subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
        ax1.set_title(f'Hystérésis de l\'aimantation \n réseau de taille L={L} - h={amplitude_h} à T={T}')
        ax1.set_box_aspect(1)
        
        ax2.plot(h_up,chi_num,'k-',linewidth=0.65,label=f'χ_L numérique (L={L})')
        ax2.plot(h_up,chi_th,'r--',linewidth=1.0,label=f'χ_L Binder (M_L={M_L:.3f})')
        ax2.set_xlim(-amplitude_h, amplitude_h)
        #ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel("h",loc='right')
        ax2.set_ylabel(r'$ χ_L(h) $',loc='top')
        ax2.set_title(f'χ_L(h) \n réseau de taille L={L} à T={T}')
        ax2.set_box_aspect(1)
        
        plt.savefig(f'χ_L(h) et Hystérésis aimentation h={amplitude_h}_L={L}_T={T}.pdf', format="pdf",bbox_inches='tight')
        plt.show()

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")