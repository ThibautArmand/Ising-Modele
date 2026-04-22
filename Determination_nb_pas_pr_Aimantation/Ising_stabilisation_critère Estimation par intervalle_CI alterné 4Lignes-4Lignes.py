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
    remplissage_alterné_4lignes_4lignes,
    remplissage_aleatoire_reseau,
    calculate_H,
    calculate_dE,
    MonteCarlo,
    format_time
)
temps_init = time.time()

Lvalues=np.array([4,8,16,32,64,128]) #valeurs de tailles du réseau : L*L
nvalues = Lvalues*Lvalues #definit un pas de Monte Carlo pour chaque taille
J=1 #constante d'échange
h=0 #champ magnétique extérieur
T_s = [0.5, 1.5, 2.5, 3.5, 5.0]#température variable

for j, L in enumerate(Lvalues):
    n=nvalues[j]
    
    fig, axes = plt.subplots(1,5,figsize=(10, 10))
    axes = axes.flatten()
    
    m=[] # aimantations pour chaques températures
    for i, T in enumerate(T_s):
        print(f"Début calul pour L={L} T={T}")
        ax = axes[i]
        Reseau = remplissage_alterné_4lignes_4lignes(L)
        m_T=np.zeros(1)
        m_T[0]=np.abs(np.mean(Reseau))
        for k in range(50):
            Reseau = MonteCarlo(n, L, Reseau, T,J,h)
            m_T=np.append(m_T,np.abs(np.mean(Reseau)))
        
        p=0
        while (p<0.95):
            for k in range(50):
                Reseau = MonteCarlo(n, L, Reseau, T,J,h)
                m_T=np.append(m_T,np.abs(np.mean(Reseau)))
            µ=np.mean(m_T[len(m_T)-50:]) # moyenne des 50 dernières valeurs
            σ=np.std(m_T[len(m_T)-50:]) #Ecart-type des 50 dernières valeurs
            m_inf=µ-1.96*σ #borne inférieur de l'intervalle
            m_sup=µ+1.96*σ #borne superieur de l'intervalle
            p=0 # p= % des 50 avant-dernières valeurs dans l'intervalle
            for k in range(50):
                if m_inf <= m_T[len(m_T)-(51+k)] <= m_sup:
                    p+=1
            p/=50
                    
        m.append(m_T)
        ax.imshow(Reseau)
        ax.set_title(f'T = {T}')
        print(f"Fin calul pour L={L} T={T}")
        
    plt.savefig(f'Configuration finale-critère Estimation par intervalle_CI Alterné 4lignes-4lignes_L={L}.pdf', bbox_inches='tight')
    plt.show()

    plt.plot(range(len(m[0])),m[0],'k-',linewidth=0.65,label=f'T={T_s[0]}')
    plt.plot(range(len(m[1])),m[1],'r-',linewidth=0.65,label=f'T={T_s[1]}')
    plt.plot(range(len(m[2])),m[2],'y-',linewidth=0.65,label=f'T={T_s[2]}')
    plt.plot(range(len(m[3])),m[3],'b-',linewidth=0.65,label=f'T={T_s[3]}')
    plt.plot(range(len(m[4])),m[4],'k--',linewidth=0.65,label=f'T={T_s[4]}')
    #plt.plot(range(len(m[5])),m[5],'-',color='orange',linewidth=0.65,label=f'T={T_s[5]}')
    plt.xlabel("nb pas",loc='right')
    plt.ylabel(r'$ | m | $',loc='top')
    plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
    plt.title(f'Evolution selon critère Estimation par intervalle,Condition Initiale alterné 4lignes-4lignes, Réseau taille L={L}')
    plt.savefig(f'Evolution-critère Estimation par intervalle_CI Alterné 4lignes-4lignes_L={L}.pdf', format="pdf",bbox_inches='tight')
    plt.show()

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")
