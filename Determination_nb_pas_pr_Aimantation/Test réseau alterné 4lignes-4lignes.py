# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:33:06 2026

@author: thiba
"""

import numpy as np
import matplotlib.pyplot as plt

L=32

X=np.zeros((L,L))
for i in range(L):
        if ((i//4)%2)==0:
            X[:][i]=1
        else:
            X[:][i]=-1