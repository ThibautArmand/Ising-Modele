# -*- coding: utf-8 -*-
"""
Modèle d'Ising : Transition de phase du premier ordre
Calcul de la susceptibilité χ_L(H) en fonction du champ magnétique H

Implémentation de la formule de Binder pour les transitions du premier ordre:
    χ_L(H) = χ_L^D + (β M_L L^d) / cosh²(β H M_L L^d)
    
où:
    - χ_L^D = 1/T est la susceptibilité "déconnectée" (réponse paramagnétique)
    - β = 1/(k_B T) = 1/T (avec k_B = 1)
    - M_L = aimantation spontanée à H=0
    - d = 2 (dimension du réseau carré)
    - L = taille du système

Référence: K. Binder, Phys. Rev. Lett. 47, 693 (1981)
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
    format_time,
    theoretical_magnetization
)
temps_init = time.time()

Lvalues=np.array([4,6,8,32]) #taille du réseau : L*L
n = Lvalues*Lvalues #definit un pas de MonteCarlo
J=1 #constante d'échange

#champ magnétique extérieur
amplitude_h=5
taille_h=200
h=np.arange(-amplitude_h,amplitude_h,2*amplitude_h/taille_h)
h=np.append(h,-h)
h=np.append(h,h)

T_s = [0.5,1.5,2.0,5]#température variable

for i, T in enumerate(T_s):
    #m=[] # aimantations pour chaque taille L
    for j, L in enumerate(Lvalues):
        Reseau = remplissage_aleatoire_reseau(L)
        m_L=np.zeros(len(h))
        
        for k in range(len(h)):
            m_L[k]=np.mean(Reseau)
            Reseau = MonteCarlo(n[j], L, Reseau, T,J,h[k])
        
        # Calculate numerical susceptibility from magnetization curve
        χ_L=np.gradient(m_L[taille_h:3*taille_h], h[taille_h:3*taille_h], edge_order=1)
        
        # Calculate theoretical susceptibility using Binder formula for first-order transitions
        # χ_L(H) = χ_L^D + (β M_L L^d) / cosh²(β H M_L L^d)
        # where χ_L^D = 1/T (disconnected susceptibility)
        #       β = 1/(k_B T) = 1/T (with k_B = 1)
        #       d = 2 (square lattice)
        #       M_L = spontaneous magnetization at H=0
        
        # Estimate M_L from theoretical formula or saturation value
        Tc = 2.269  # Critical temperature for 2D Ising model
        if T < Tc:
            # Below Tc: use theoretical spontaneous magnetization
            M_L = theoretical_magnetization(np.array([T]), Tc, J)[0]
        else:
            # Above Tc: use saturation magnetization from the hysteresis loop
            M_L = np.abs(np.mean([m_L[taille_h], m_L[2*taille_h-1]]))
        
        # Binder formula for susceptibility
        beta = 1.0 / T  # β = 1/(k_B T) with k_B = 1
        d = 2  # dimension of square lattice
        h_range = h[taille_h:3*taille_h]
        
        chi_D = beta  # Disconnected susceptibility χ_L^D = 1/T
        chi_connected = (beta * M_L * L**d) / np.cosh(beta * h_range * M_L * L**d)**2
        χ_Lth = chi_D + chi_connected
        
        fig , (ax1,ax2) = plt.subplots(1,2,figsize=(10, 10))
        ax1.plot(h[taille_h:2*taille_h],m_L[taille_h:2*taille_h],'k-',linewidth=0.65,label=f'h de {h[taille_h]:.1f} à {h[2*taille_h-1]:.1f}')
        ax1.plot(h[2*taille_h:3*taille_h],m_L[2*taille_h:3*taille_h],'r-',linewidth=0.65,label=f'h de {h[2*taille_h]:.1f} à {h[3*taille_h-1]:.1f}')
        ax1.set_xlim(-amplitude_h, amplitude_h)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlabel("h",loc='right')
        ax1.set_ylabel(r'$ m $',loc='top')
        ax1.legend()
        ax1.set_title(f'Hystérésis de l\'aimantation \n réseau de taille L={L} - h={amplitude_h} à T={T}')
        ax1.set_box_aspect(1)
        
        ax2.plot(h[taille_h:3*taille_h],χ_L,'k-',linewidth=0.65,label=f'χ_L numérique')
        ax2.plot(h[taille_h:3*taille_h],χ_Lth,'r--',linewidth=1.0,label=f'χ_L Binder (M_L={M_L:.3f})')
        ax2.set_xlim(-amplitude_h, amplitude_h)
        #ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel("h",loc='right')
        ax2.set_ylabel(r'$ χ_L(h) $',loc='top')
        ax2.set_title(f'χ_L(h) - Formule de Binder\n réseau de taille L={L} à T={T}')
        ax2.legend()
        ax2.set_box_aspect(1)
        
        plt.savefig(f'χ_L(h) et Hystérésis aimentation h={amplitude_h}_L={L}_T={T}.pdf', format="pdf",bbox_inches='tight')
        plt.show()

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")