#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:33:47 2018

@author: cardoso
"""

import airfoil
import mesh
import matplotlib.pyplot as plt
import numpy as np

# nombre de archivo perfil externo
filename = 'whitcomb-il.txt'

# densidad de puntos perfil NACA - Debe ser Impar
points = 109
# se ajusta por cuestiones de calculo a la mitad de puntos, se calculan por separado parte inferior y superior
points = (points + 1) // 2

# densidad del mallado
M = (points * 2) - 1
N = M // 2

# datos de perfil NACA
m = 0 # combadura
p = 0 # posicion de la combadura
t = 24 # espesor
c = 1 # cuerda [m]

# radio frontera externa
R = 10 * c


perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
#perfil.plot()
del(perfil)

# nombre de archivo con perfil redimensionado y con c/4 en el origen del sistema de coordenadas
archivo_perfil = 'perfil_final.txt'
mallaNACA = mesh.mesh_O(R, M, N, archivo_perfil)


plt.figure('NACA')
plt.title('Interpolación Polinomial')
mallaNACA.gen_inter_pol()
mallaNACA.plot()

plt.figure('NACA_otro')
plt.title('Interpolación Hermite')
(X,Y) = mallaNACA.gen_inter_Hermite()
mallaNACA.plot()

plt.figure('otra prueba')
plt.plot(X, Y, 'k', linewidth = 0.5)
plt.axis('equal')
size = np.size(X, 0)
for i in range(size // 2 - 3, size//2 + 3):
    plt.plot(X[i, :], Y[i, :], linewidth = 1.3)
for i in range(-2, 2):
    plt.plot(X[i, :], Y[i, :])



