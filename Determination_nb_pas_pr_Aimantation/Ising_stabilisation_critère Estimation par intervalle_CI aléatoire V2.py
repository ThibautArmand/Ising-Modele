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

Lvalues=np.array([16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]) #valeurs de tailles du réseau : L*L
nvalues = Lvalues*Lvalues #definit un pas de Monte Carlo pour chaque taille
J=1 #constante d'échange
h=0 #champ magnétique extérieur
T_s = [0.5,1.5,2.5,3.5]#température variable

nb_pas=np.zeros((len(T_s),len(Lvalues))) #tableau du nb de pas=[T:indice i][L:indice j]
         
for i, T in enumerate(T_s):
    for j, L in enumerate(Lvalues):
        print(f"Début calul pour T={T} L={L} ")
        n=nvalues[j]
        m=[] # aimantation à chaque pas Monte Carlo
        m_f=[] # aimantation finale du réseau à chaque itération s
        for s in range(25):
            Reseau = remplissage_aleatoire_reseau(L)
            m_L=np.zeros(1)
            m_L[0]=np.abs(np.mean(Reseau))
            for k in range(50):
                Reseau = MonteCarlo(n, L, Reseau, T,J,h)
                m_L=np.append(m_L,np.abs(np.mean(Reseau)))
            
            stab=0 
            while (stab==0): #test stabilité atteinte: vrai=1;faux=0
                for k in range(50):
                    Reseau = MonteCarlo(n, L, Reseau, T,J,h)
                    m_L=np.append(m_L,np.abs(np.mean(Reseau)))
                µ=np.mean(m_L[len(m_L)-49:]) # moyenne des 50 dernières valeurs
                σ=np.std(m_L[len(m_L)-49:]) #Ecart-type des 50 dernières valeurs
                m_inf=µ-2.576/np.sqrt(50)*σ #borne inférieur de l'intervalle
                m_sup=µ+2.576/np.sqrt(50)*σ #borne superieur de l'intervalle
                µ1=np.mean(m_L[len(m_L)-99:len(m_L)-50]) # moyenne des 100 à 50 dernières valeurs
                if m_inf <= µ1 <= m_sup :
                    stab=1
                else:
                    stab=0                   
            m.append(m_L)
            m_f.append(m_L[-1])
        
        µ_m_f=np.mean(m_f) # valeur moyenne des aimantations finales
        σ_m_f=np.std(m_f) # Ecart type
        
        index=[]
        for l in range(len(m_f)):
            if  m_f[l] < µ_m_f-σ_m_f or m_f[l] > µ_m_f+σ_m_f:
                index=np.append(l,index)
        for o in range(len(index)):
            del m[int(index[o])]
        
        nb_pas[i][j]=np.mean([len(liste) for liste in m])

        print(f"Fin calul pour T={T} L={L} ")

plt.plot(Lvalues,nb_pas[0],'b:v',linewidth=0.65,label=f'T={T_s[0]}')
plt.plot(Lvalues,nb_pas[1],'r:o',linewidth=0.65,label=f'T={T_s[1]}')
plt.plot(Lvalues,nb_pas[2],'y:s',linewidth=0.65,label=f'T={T_s[2]}')
plt.plot(Lvalues,nb_pas[3],'k:p',linewidth=0.65,label=f'T={T_s[3]}')
plt.xlabel("L",loc='right')
plt.ylabel("nb pas",rotation=0,loc='top')
plt.subplot(111).legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.15))
plt.title('Estimation du nombre de pas Monte Carlo à effectuer pour atteindre la stabilisation \n Condition Initiale aléatoire')
plt.savefig('Estimation_pas MonteCarlo_CI Aléatoire.pdf', format="pdf",bbox_inches='tight')
plt.show()

"""
            result=filter(lambda m[i][]: m[i][-1] < µ_m_f-σ_m_f or m[i][-1] > µ_m_f+σ_m_f, m)

            for i in range(len(m_f)):
                if  m[i][-1] < µ_m_f-σ_m_f or m[i][-1] > µ_m_f+σ_m_f:
                    m.pop(i)
            
            fig , (ax1,ax2) = plt.subplots(1,2,figsize=(10, 10))
            ax1.plot(range(len(m_T)),m_T,'k-',linewidth=0.65,label=f'T={T_s[0]}')
            ax1.set_xlabel("nb pas",loc='right')
            ax1.set_ylabel(r'$ | m | $',loc='top')
            ax1.set_title(f'Courbe aimantation \n Condition Initiale aléatoire, réseau de taille L={L} T={T}')
            ax1.set_box_aspect(1)
            
            ax2.imshow(Reseau)
            ax2.set_title(f'Configuration finale du réseau : L={L} T = {T}')
            ax1.set_box_aspect(1)
            
            plt.savefig(f'Courbe Aimantation_CI Aléatoire_T={T}_L={L} n°{s}.pdf', format="pdf",bbox_inches='tight')
            plt.show()
"""

temps_end = time.time()
temps_diff = temps_end - temps_init
print(f"Temps d'exécution : {format_time(temps_diff)}")
