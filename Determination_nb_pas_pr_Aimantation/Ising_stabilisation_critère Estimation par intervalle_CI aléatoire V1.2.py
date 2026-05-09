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

Lvalues=np.array([16,32,64,128]) #valeurs de tailles du réseau : L*L
nvalues = Lvalues*Lvalues #definit un pas de Monte Carlo pour chaque taille
J=1 #constante d'échange
h=0 #champ magnétique extérieur
T_s = [0.5, 2.0, 3.5, 5.0]#températures du réseau
Echantillon=50
for j, L in enumerate(Lvalues):
    n=nvalues[j]
    
    fig, axes = plt.subplots(1,len(T_s),figsize=(10, 10))
    axes = axes.flatten()
    
    m=[] # aimantations pour chaque température
    nb_pas=np.zeros(len(T_s)) #nb pas pour stabilisation à chaque température
    for i, T in enumerate(T_s):
        print(f"Début calul pour L={L} T={T}")
        ax = axes[i]
        Reseau = remplissage_aleatoire_reseau(L)
        m_T=np.zeros(1)
        m_T[0]=np.abs(np.mean(Reseau))
        for k in range(3*Echantillon-1):
            Reseau = MonteCarlo(n, L, Reseau, T,J,h)
            m_T=np.append(m_T,np.abs(np.mean(Reseau)))
        
        stab=0 #test stabilité atteinte: vrai=1;faux=0
        while (stab==0):
            µ=np.mean(m_T[len(m_T)-(Echantillon-1):]) # moyenne des 50 dernières valeurs
            σ=np.std(m_T[len(m_T)-(Echantillon-1):]) #Ecart-type des 50 dernières valeurs
            m_inf=µ-2.576/np.sqrt(Echantillon)*σ #borne inférieur de l'intervalle
            m_sup=µ+2.576/np.sqrt(Echantillon)*σ #borne superieur de l'intervalle
            
            µ1=np.mean(m_T[len(m_T)-(2*Echantillon):len(m_T)-Echantillon-1]) # moyenne des 100 à 50 dernières valeurs
                        
            if m_inf <= µ1 <= m_sup :
                µ2=np.mean(m_T[len(m_T)-(3*Echantillon):len(m_T)-2*Echantillon-1])
                if m_inf <= µ2 <= m_sup :
                    stab=1
                else:
                    for k in range(25):
                        Reseau = MonteCarlo(n, L, Reseau, T,J,h)
                        m_T=np.append(m_T,np.abs(np.mean(Reseau)))
            else:
                for k in range(25):
                    Reseau = MonteCarlo(n, L, Reseau, T,J,h)
                    m_T=np.append(m_T,np.abs(np.mean(Reseau)))
                
        nb_pas[i]=len(m_T)-3*Echantillon
        while (len(m_T)<2000):
            Reseau = MonteCarlo(n, L, Reseau, T,J,h)
            m_T=np.append(m_T,np.abs(np.mean(Reseau)))
        
        m.append(m_T)
        ax.imshow(Reseau)
        ax.set_title(f'T = {T}')
        print(f"Fin calul pour L={L} T={T}")
        
    plt.savefig(f'V1-2_Configuration finale-critère Estimation par intervalle_CI Aléatoire_L={L}.pdf', bbox_inches='tight')
    plt.show()

    plt.plot(range(len(m[0])),m[0],'k-',linewidth=0.65,label=f'T={T_s[0]}')
    plt.axvline(nb_pas[0],0,1,color='k',linewidth=0.65,ls='--')
    plt.plot(range(len(m[1])),m[1],'r-',linewidth=0.65,label=f'T={T_s[1]}')
    plt.axvline(nb_pas[1],0,1,color='r',linewidth=0.65,ls=':')
    plt.plot(range(len(m[2])),m[2],'y-',linewidth=0.65,label=f'T={T_s[2]}')
    plt.axvline(nb_pas[2],0,1,color='y',linewidth=0.65,ls='-.')
    plt.plot(range(len(m[3])),m[3],'b-',linewidth=0.65,label=f'T={T_s[3]}')
    plt.axvline(nb_pas[3],0,1,color='b',linewidth=0.65,ls='-')
    
    plt.xlabel("nb pas",loc='right')
    plt.ylabel(r'$ | m | $',loc='top')
    plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
    plt.title(f'Evolution selon critère Estimation par intervalle,Condition Initiale aléatoire de taille L={L}')
    plt.savefig(f'V1-2_Evolution-Estimation par intervalle_CI Aléatoire_L={L}.pdf', format="pdf",bbox_inches='tight')
    plt.show()

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")
