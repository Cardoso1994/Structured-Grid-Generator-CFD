#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:33:47 2018

@author: cardoso
"""

import numpy as np
import matplotlib.pyplot as plt

import airfoil
import mesh
import mesh_c
import mesh_o
import mesh_su2
from potential import potential_flow_o, potential_flow_o_esp
import helpers

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
N = 35
union = 6

# points = 11
airfoil_points = 35

if malla == 'C':
    points = airfoil_points // 3 * 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 0  # combadura
p = 0  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]
# radio frontera externa
R = 20 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
# perfil.join(flap, dx=0.055, dy=0.05, union=union)
# perfil.rotate(30)
# M = np.shape(perfil.x)[0]

archivo_perfil = 'perfil_final.csv'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

mallaNACA.X[:, 0] += 0.25
mallaNACA.Y[0, :] = 0
mallaNACA.Y[-1, :] = 0
mallaNACA.gen_Laplace()
# mallaNACA.X = np.flip(mallaNACA.X)
# mallaNACA.Y = np.flip(mallaNACA.Y)

print('after mesh generation')
print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

# variables de flujo
t_inf = 300
p_inf = 101300
v_inf = 10

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
(phi, C, theta, IMA) = potential_flow_o_esp(d0, h0, gamma, mach_inf, v_inf, alfa, mallaNACA)

plt.figure('potential')
plt.contour(mallaNACA.X, mallaNACA.Y, phi, 50)
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
plt.axis('equal')

plt.draw()
plt.show()
