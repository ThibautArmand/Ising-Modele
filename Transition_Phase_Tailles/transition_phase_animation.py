# -*- coding: utf-8 -*-
"""
Modèle d'Ising - Animation de l'évolution du réseau en fonction de la température
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.utils import (
    remplissage_aleatoire_reseau,
    theoretical_magnetization,
    MonteCarlo,
    simulate_temperature,
    format_time
)

# Paramètres
L = 64  # Taille du réseau
Tc = 2.269  # Température critique
T_min = 1.0
T_max = 4.0
steps_T = 50
n_equilibration = 10000 
n_measurements = 1000
J = 1.0
h = 0.0
time_init = time.time()
T_s = np.linspace(T_min, T_max, steps_T) 

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)

ax_lattice = fig.add_subplot(gs[:, 0])
Reseau = remplissage_aleatoire_reseau(L)

im = ax_lattice.imshow(Reseau, vmin=-1, vmax=1, interpolation='nearest')
title_text = ax_lattice.set_title(f'T = {T_s[0]:.3f} K', fontsize=16)
ax_lattice.axis('off')
cbar = plt.colorbar(im, ax=ax_lattice, fraction=0.046, pad=0.04)
cbar.set_label('Spin', fontsize=12)

# 1. Énergie par spin
ax_energy = fig.add_subplot(gs[0, 1])
ax_energy.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
ax_energy.set_ylabel('$\\langle e \\rangle$', fontsize=12)
ax_energy.set_title('Énergie par spin', fontsize=13)
ax_energy.set_xlim(T_s[0] - 0.1, T_s[-1] + 0.1)
ax_energy.set_ylim(-4.1, 0.2)
ax_energy.legend(fontsize=10)
ax_energy.grid(True, alpha=0.3)
energy_marker, = ax_energy.plot([], [], 'o', color='black', markersize=8, zorder=5)
energy_line, = ax_energy.plot([], [], '-', color='black', linewidth=2, alpha=0.8)

# 2. Aimantation par spin
ax_mag = fig.add_subplot(gs[1, 1])
ax_mag.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
ax_mag.set_ylabel('$\\langle |m| \\rangle$', fontsize=12)
ax_mag.set_title('Aimantation par spin', fontsize=13)
# Courbe théorique
m_theo = theoretical_magnetization(T_s, Tc)
ax_mag.plot(T_s, m_theo, 'k--', linewidth=2, label='Théorie', alpha=0.5)
ax_mag.set_xlim(T_s[0] - 0.1, T_s[-1] + 0.1)
ax_mag.set_ylim(0, 1.05)
ax_mag.legend(fontsize=10)
ax_mag.grid(True, alpha=0.3)
mag_marker, = ax_mag.plot([], [], 'o', color='black', markersize=8, zorder=5)
mag_line, = ax_mag.plot([], [], '-', color='black', linewidth=2, alpha=0.8)

# 3. Chaleur spécifique
ax_heat = fig.add_subplot(gs[2, 1])
ax_heat.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
ax_heat.set_ylabel('$C$', fontsize=12)
ax_heat.set_title('Chaleur spécifique', fontsize=13)
ax_heat.set_xlim(T_s[0] - 0.1, T_s[-1] + 0.1)
ax_heat.set_ylim(0, 6)
ax_heat.legend(fontsize=10)
ax_heat.grid(True, alpha=0.3)
heat_marker, = ax_heat.plot([], [], 'o', color='black',  markersize=8, zorder=5)
heat_line, = ax_heat.plot([], [], '-', color='black', linewidth=2, alpha=0.8)

# 4. Susceptibilité magnétique
ax_chi = fig.add_subplot(gs[3, 1])
ax_chi.axvline(Tc, color='red', linestyle='--', linewidth=2, label=f'$T_c$ = {Tc:.3f}')
ax_chi.set_xlabel('Température (K)', fontsize=12)
ax_chi.set_ylabel('$\\chi$', fontsize=12)
ax_chi.set_title('Susceptibilité magnétique', fontsize=13)
ax_chi.set_xlim(T_s[0] - 0.1, T_s[-1] + 0.1)
ax_chi.set_ylim(0, 100)
ax_chi.legend(fontsize=10)
ax_chi.grid(True, alpha=0.3)
chi_marker, = ax_chi.plot([], [], 'o', color='black', markersize=8, zorder=5)
chi_line, = ax_chi.plot([], [], '-', color='black', linewidth=2, alpha=0.8)

# Stockage des données
temperatures_done = []
energies = []
magnetizations = []
specific_heats = []
susceptibilities = []

def animate(frame):
    global Reseau
    T = T_s[frame]
    N = L**2
    
    e, m, C, chi = simulate_temperature(L, T, n_equilibration, n_measurements, J, h)

    for _ in range(n_equilibration // 10):
        Reseau = MonteCarlo(N, L, Reseau, T, J, h)
    
    im.set_array(Reseau)
    title_text.set_text(f'T = {T:.3f} K (Frame {frame+1}/{len(T_s)})')
    
    temperatures_done.append(T)
    energies.append(e)
    magnetizations.append(m)
    specific_heats.append(C)
    susceptibilities.append(chi)
    
    # Updating...
    # 1. Énergie
    energy_marker.set_data([T], [e])
    energy_line.set_data(temperatures_done, energies)
    if len(energies) > 1:
        ax_energy.set_ylim(min(energies) * 1.1, max(energies) * 1.1)
    
    # 2. Aimantation
    mag_marker.set_data([T], [m])
    mag_line.set_data(temperatures_done, magnetizations)
    
    # 3. Chaleur spécifique
    heat_marker.set_data([T], [C])
    heat_line.set_data(temperatures_done, specific_heats)
    if len(specific_heats) > 1:
        ax_heat.set_ylim(0, max(specific_heats) * 1.1)
    
    # 4. Susceptibilité
    chi_marker.set_data([T], [chi])
    chi_line.set_data(temperatures_done, susceptibilities)
    if len(susceptibilities) > 1:
        ax_chi.set_ylim(0, max(susceptibilities) * 1.1)
    
    if (frame + 1) % 5 == 0 or frame == 0:
        print(f"  Frame {frame+1}/{len(T_s)} - T = {T:.3f} K")
    
    return [im, title_text, energy_marker, energy_line, mag_marker, mag_line, 
            heat_marker, heat_line, chi_marker, chi_line]

output_file = f'../results/phase_transition_animation_{T_min}_{T_max}_{steps_T}_{int(time_init)}.mp4'
print(f"\nStarting animation {output_file} ...")

anim = animation.FuncAnimation(
    fig, animate, frames=len(T_s), 
    interval=100,  # 100 ms entre les frames lors de l'affichage
    blit=True, 
    repeat=True,
)

try:
    anim.save(output_file, 
              writer='ffmpeg', 
              fps=10,  # 10 frames par seconde
              dpi=150,
              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    print(f"\n✓ Finished :D {output_file}")
except Exception as e:
    print(f"\n✗ Error D: {e}")
plt.show()

time_end = time.time()
time_diff = time_end - time_init
print(f"\nTemps total d'exécution : {format_time(time_diff)}")
