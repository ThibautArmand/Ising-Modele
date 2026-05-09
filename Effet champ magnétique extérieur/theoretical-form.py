#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:59:52 2026

@author: carguello
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
output_dir = Path(__file__).parent.parent / "results" / "effet-champ-magnetique-exterieur"

import numpy as np
import matplotlib.pyplot as plt
from utils.utils import (
    theoretical_magnetization,
)

J = 1.0
T = 2.1
Beta = 1/T
Lvalues = np.array([4, 6, 8, 10, 12, 14])
x = np.linspace(0, 8, 300)  # x = H L^2 / J

Msp = float(theoretical_magnetization(np.array([T]), J=J)[0])

plt.figure(figsize=(6, 5))

for L in Lvalues:
    # scaling
    # x = H L^d / J -> H L^d = x J ->(J=1)-> H L^d = x
    y = (Msp**2 * Beta) / np.cosh(x * Msp * Beta)**2
    plt.plot(x, y, label=f"L={L}")

plt.xlabel(r"$HL^2/J$")
plt.ylabel(r"$\chi_L/L^2$")
plt.title(r"Scaling de Binder-Landau, $T=2.1$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / f"theoretical_binder_landau_T={T}.pdf", format="pdf", bbox_inches='tight')
plt.show()