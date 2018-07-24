#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:33:47 2018

@author: cardoso
"""

import airfoil
import mesh
import matplotlib.pyplot as plt
# import numpy as np

# nombre de archivo perfil externo
filename = 'whitcomb-il.txt'

# tipo de malla (H, C, O)
malla = 'C'

# densidad de puntos perfil NACA - Debe ser Impar
points = 67
# se ajusta por cuestiones de calculo a la mitad de puntos, se calculan por separado parte inferior y superior
points = (points + 1) // 2


# datos de perfil NACA
m = 2 # combadura
p = 4 # posicion de la combadura
t = 12# espesor
c = 1 # cuerda [m]
# radio frontera externa
R = 5 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
#perfil.plot()
del(perfil)

# se ajusta la densidad de la malla dependiendo el tipo de malla a generar
# densidad del mallado normal si es tipo 'O'
M = (points * 2) - 1
if malla == 'C':
    M += 3 * M // 4
elif malla == 'H':
    M *= 2
N = 45




# nombre de archivo con perfil redimensionado y con c/4 en el origen del sistema de coordenadas
archivo_perfil = 'perfil_final.txt'
if malla == 'O':
    mallaNACA = mesh.mesh_O(R, M, N, archivo_perfil)
elif malla == 'C':
    mallaNACA = mesh.mesh_C(R, M, N, archivo_perfil)
elif malla == 'H':
    mallaNACA = mesh.mesh_H(R, M, N, archivo_perfil)


'''
(X,Y) = mallaNACA.gen_inter_Hermite()
plt.figure('NACA')
plt.title('Interpolación Hermite')
mallaNACA.plot()
'''

'''
mallaNACA.gen_inter_pol()
plt.figure('NACA_')
plt.title('Interpolación Polinomial')
mallaNACA.plot()
'''


mallaNACA.gen_EDP(ec = 'P', metodo = 'SOR')
plt.figure('_NACA_')
plt.title('Ec de Poisson')
mallaNACA.plot()


'''
mallaNACA.gen_EDP(ec = 'L', metodo = 'GS')
plt.figure('_NACA_Laplace')
plt.title('EDc de Laplace')
mallaNACA.plot()
'''

'''
mallaNACA.gen_TFI()
plt.figure('___NACA_')
plt.title('Interpolación TFI')
mallaNACA.plot()
'''