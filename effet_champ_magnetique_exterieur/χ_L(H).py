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
    simulate_chi_at_H,
    format_time
)
temps_init = time.time()

Lvalues=np.array([4,6,8,10,12,14]) #taille du réseau : L*L
J=1 #constante d'échange
d = 2

#champ magnétique extérieur
amplitude_h=5
taille_h=10
h_up=np.linspace(-amplitude_h,amplitude_h,taille_h)
h_down=np.linspace(amplitude_h,-amplitude_h,taille_h)
h=np.concatenate([h_up, h_down])

T_s = [0.5,1.5,2.0,5]#température variable

for i, T in enumerate(T_s):
    beta = 1/T
    data_all_L = {}
    for j, L in enumerate(Lvalues):
        Reseau = remplissage_aleatoire_reseau(L)
        m_L  = np.zeros(len(h))
        chi_L = np.zeros(len(h))

        for k in range(len(h)):
            Reseau, m_L[k], chi_L[k] = simulate_chi_at_H(
                Reseau, L, T, J, h[k], n_equilib=1000, n_measure=500
            )

        m_up   = m_L[:taille_h]
        m_down = m_L[taille_h:]
        chi_up   = chi_L[:taille_h]
        chi_down = chi_L[taille_h:]
        
        data_all_L[L] = {
            'm_up': m_up, 'm_down': m_down,
            'chi_up': chi_up, 'chi_down': chi_down
        }

        idx_zero_up = np.argmin(np.abs(h_up))
        idx_zero_down = np.argmin(np.abs(h_down))
        
        M_L_up = np.abs(m_up[idx_zero_up])
        M_L_down = np.abs(m_down[idx_zero_down])

        chi_D = 0.0
        chi_th_up = chi_D + (beta * M_L_up**2 * L**d) / (np.cosh(beta * h_up * M_L_up * L**d))**2
        chi_th_down = chi_D + (beta * M_L_down**2 * L**d) / (np.cosh(beta * h_down * M_L_down * L**d))**2

        fig , (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10, 10))
        ax1.plot(h_up,m_up,'k-',linewidth=0.65,label=f'h croissant: {h_up[0]:.2f} à {h_up[-1]:.2f}')
        ax1.plot(h_down,m_down,'r-',linewidth=0.65,label=f'h décroissant: {h_down[0]:.2f} à {h_down[-1]:.2f}')
        ax1.set_xlim(-amplitude_h, amplitude_h)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlabel("h",loc='right')
        ax1.set_ylabel(r'$ m $',loc='top')
        ax1.set_title(f'Hystérésis de l\'aimantation \n réseau de taille L={L} - h={amplitude_h} à T={T}', fontsize=8)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=7)
        ax1.set_box_aspect(1)
        
        ax2.plot(h_up,chi_up,'k-',linewidth=0.65,label=f'χ_L variance (L={L})')
        ax2.plot(h_up,chi_th_up,'r--',linewidth=1.0,label=f'χ_L Binder (M_L={M_L_up:.3f})')
        ax2.set_xlim(-amplitude_h, amplitude_h)
        ax2.set_xlabel("h",loc='right')
        ax2.set_ylabel(r'$ χ_L(h) $',loc='top')
        ax2.set_title(f'χ_L(h) \n réseau de taille L={L} à T={T}', fontsize=8)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=7)
        ax2.set_box_aspect(1)

        ax3.plot(h_down,chi_down,'k-',linewidth=0.65,label=f'χ_L variance (L={L})')
        ax3.plot(h_down,chi_th_down,'r--',linewidth=1.0,label=f'χ_L Binder (M_L={M_L_down:.3f})')
        ax3.set_xlim(-amplitude_h, amplitude_h)
        ax3.set_xlabel("h",loc='right')
        ax3.set_ylabel(r'$ χ_L(h) $',loc='top')
        ax3.set_title(f'χ_L(h) \n réseau de taille L={L} à T={T}', fontsize=8)
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=7)
        ax3.set_box_aspect(1)
        
        plt.savefig(f'χ_L(h) et Hystérésis aimentation h={amplitude_h}_L={L}_T={T}.pdf', format="pdf",bbox_inches='tight')
        plt.show()
    
    # First Order Scaling
    # χ_L / L² vs H L² / J pour tous les L
    fig_scaling, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for idx, L in enumerate(Lvalues):
        data = data_all_L[L]
        
        chi_scaled_down = data['chi_down'] / (L**2)
        h_scaled_down = h_down * (L**2) / J
        
        # x > 0
        mask_down = h_scaled_down > 0
        
        ax.plot(h_scaled_down[mask_down], chi_scaled_down[mask_down], 
                     marker='o', linewidth=1.0, markersize=5,
                     label=f'L={L}')
    
    ax.set_xlabel(r'$H L^2 / J$', fontsize=12)
    ax.set_ylabel(r'$\chi_L / L^2$', fontsize=12)
    ax.set_title(f'First Order Scaling\nT={T}', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'First_Order_Scaling_T={T}.pdf', format="pdf", bbox_inches='tight')
    plt.show()

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")