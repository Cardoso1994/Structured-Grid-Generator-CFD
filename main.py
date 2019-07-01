#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:33:47 2018

@author: cardoso
"""


import airfoil
import mesh
import mesh_c
import mesh_o
import numpy as np


import matplotlib.pyplot as plt

# nombre de archivo perfil externo
filename = 'whitcomb-il.txt'

# tipo de malla (C, O)
malla = 'C'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
M = 121
N = 45
# se ajusta por cuestiones de calculo a la mitad de puntos
# se calculan por separado parte inferior y superior del perfil
# points = (M + 1) // 2

if malla == 'C':
    points = M // 3 * 2
elif malla == 'O':
    points = M
# datos de perfil NACA
m = 2  # combadura
p = 4  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]
# radio frontera externa
R = 5 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
# se ajusta la densidad de la malla dependiendo el tipo de malla a generar
# densidad del mallado normal si es tipo 'O'
#M = (points * 2) - 1
#if malla == 'C':
    #M += 3 * M // 4
#N = 17



# nombre de archivo con perfil redimensionado y con c/4 en el origen del sistema de coordenadas
archivo_perfil = 'perfil_final.txt'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, M, N, archivo_perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, M, N, archivo_perfil)



'''mallaNACA.gen_inter_Hermite()
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


mallaNACA.gen_Poisson(metodo='SOR')
plt.figure('_NACA_')
plt.title('Ec de Poisson')
mallaNACA.plot()

mallaNACA.gen_Laplace(metodo='SOR')
plt.figure('_NACA_Laplace')
plt.title('Ec de Laplace')
mallaNACA.plot()


'''
mallaNACA.gen_TFI()
plt.figure('___NACA_')
plt.title('Interpolación TFI')
mallaNACA.plot()
'''

print(M, N)
