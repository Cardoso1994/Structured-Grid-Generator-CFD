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
from analysis import potential_flow_o, potential_flow_o_esp

import matplotlib.pyplot as plt

# nombre de archivo perfil externo
filename = 'whitcomb-il.txt'

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
M = 35
N = 35
# se ajusta por cuestiones de calculo a la mitad de puntos
# se calculan por separado parte inferior y superior del perfil
# points = (M + 1) // 2

if malla == 'C':
    points = M // 3 * 2
elif malla == 'O':
    points = M
# datos de perfil NACA
m = 1  # combadura
p = 2  # posicion de la combadura
t = 10  # espesor
c = 1  # cuerda [m]
# radio frontera externa
R = 20 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
# perfil.create_linear(points)
# flap = airfoil.NACA4(m, p, t, 0.2 * c)
# flap.create_sin(points)
# flap.rotate(15)
# perfil.join(flap, dx=0.055, dy=0.05, join_section=4)
# perfil.rotate(25)
M = np.shape(perfil.x)[0]

if malla == 'C':
    M = M // 2 * 3
    print(M)
# se ajusta la densidad de la malla dependiendo el tipo de malla a generar
# densidad del mallado normal si es tipo 'O'
'''
M = (points * 2) - 1
if malla == 'C':
   #M += 3 * M // 4
N = 17
'''


# nombre de archivo con perfil redimensionado y con c/4 en el origen del
# sistema de coordenadas
archivo_perfil = 'perfil_final.txt'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, M, N, archivo_perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, M, N, archivo_perfil)

'''
mallaNACA.gen_inter_Hermite()
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

'''
mallaNACA.gen_Poisson(metodo='SOR')
plt.figure('_NACA_')
plt.title('Ec de Poisson')
mallaNACA.plot()
'''


mallaNACA.gen_Laplace(metodo='SOR')
#  plt.figure('_NACA_Laplace')
#  plt.title('Ec de Laplace')
#  mallaNACA.plot()

mallaNACA.X = np.flip(mallaNACA.X)
mallaNACA.Y = np.flip(mallaNACA.Y)
###############################################################################
#
# Las mallas coinciden perfectamente
#
###############################################################################

'''
mallaNACA.gen_TFI()
plt.figure('___NACA_')
plt.title('Interpolación TFI')
mallaNACA.plot()
'''

'''
mallaNACA.gen_parabolic()
#plt.figure('NACA_Parabolic')
#plt.title('Parabolic gen')
mallaNACA.plot()
'''
print(M, N)


# variables de flujo
t_inf = 273.15
p_inf = 101325
v_inf = 75

alfa = 0

gamma = 1.4
cp = 1006
Rg = cp * (gamma - 1) / gamma
d_inf = p_inf / (Rg * t_inf)
h_inf = cp * t_inf
c_inf = (gamma * p_inf / d_inf) ** 0.5

h0 = h_inf + 0.5 * v_inf ** 2
d0 = d_inf / (1 - 0.5 * v_inf ** 2 / h0)
p0 = p_inf * (d0 / d_inf) ** gamma

mach_inf = v_inf / c_inf
Re = v_inf * c * d_inf / 17e-6
print(mach_inf)
print(Re)
potential_flow_o_esp(d0, h0, gamma, mach_inf, v_inf, alfa, mallaNACA)

