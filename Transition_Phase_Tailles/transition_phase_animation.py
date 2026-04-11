# -*- coding: utf-8 -*-
"""
Modèle d'Ising - Animation de l'évolution du réseau en fonction de la température
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.utils import (
    remplissage_aleatoire_reseau,
    calculate_magnetization,
    MonteCarlo
)

# Paramètres
L = 64  # Taille du réseau
T_s = np.linspace(1.0, 4.0, 10) 
n_equilibration = 5000  # Pas d'équilibration par température
J = 1.0
h = 0.0
Tc = 2.269  # Température critique

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
Reseau = remplissage_aleatoire_reseau(L)

im = ax1.imshow(Reseau, vmin=-1, vmax=1, interpolation='nearest')
title_text = ax1.set_title(f'T = {T_s[0]:.3f} K', fontsize=16, fontweight='bold')
ax1.axis('off')
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Spin', fontsize=12)

ax2.axvline(Tc, color='red', linestyle='--', linewidth=2.5, label=f'$T_c$ = {Tc:.3f} K', alpha=0.7)
ax2.set_xlim(T_s[0] - 0.1, T_s[-1] + 0.1)
ax2.set_ylim(0, 1.05)
ax2.set_xlabel('Température (K)', fontsize=14)
ax2.set_ylabel('$|m|$ (Aimantation par spin)', fontsize=14)
ax2.set_title('Évolution de l\'aimantation', fontsize=16, fontweight='bold')
temp_marker, = ax2.plot([], [], 'o', color='darkblue', markersize=12, 
                         markeredgecolor='white', markeredgewidth=2, zorder=5)
mag_line, = ax2.plot([], [], '-', color='darkblue', linewidth=3, alpha=0.8)
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

temperatures_done = []
magnetizations = []

def animate(frame):
    global Reseau
    T = T_s[frame]
    
    # Équilibration à cette température
    for _ in range(n_equilibration):
        Reseau = MonteCarlo(L**2, L, Reseau, T, J, h)
    
    M = calculate_magnetization(Reseau)
    m = np.abs(M) / (L**2)
    
    # Mise à jour du réseau
    im.set_array(Reseau)
    title_text.set_text(f'T = {T:.3f} K (Frame {frame+1}/{len(T_s)})')
    
    # Mise à jour de la courbe d'aimantation
    temperatures_done.append(T)
    magnetizations.append(m)
    temp_marker.set_data([T], [m])
    mag_line.set_data(temperatures_done, magnetizations)
    
    if (frame + 1) % 10 == 0:
        print(f"  Frame {frame+1}/{len(T_s)} - T = {T:.3f} K, |m| = {m:.4f}")
    
    return [im, title_text, temp_marker, mag_line]

anim = animation.FuncAnimation(
    fig, animate, frames=len(T_s), 
    interval=100,  # 100 ms entre les frames lors de l'affichage
    blit=True, 
    repeat=True
)

output_file = '../results/phase_transition_animation.mp4'
print(f"\nSauvegarde de l'animation vers {output_file}...")
print("Cela peut prendre quelques minutes... :waiting:")

try:
    anim.save(output_file, 
              writer='ffmpeg', 
              fps=10,  # 10 frames par seconde
              dpi=150,
              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    print(f"\n✓ Animation sauvegardée avec succès: {output_file}")
except Exception as e:
    print(f"\n✗ Erreur lors de la sauvegarde: {e}")
plt.show()
