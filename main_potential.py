#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:33:47 2018

@author: cardoso
"""

from os import mkdir
import numpy as np
import matplotlib.pyplot as plt

import airfoil
import mesh
import mesh_c
import mesh_o
import mesh_su2
from potential import potential_flow_o, potential_flow_o_esp, velocity, pressure, lift_n_drag, streamlines
import helpers

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''

N = 65
airfoil_points = 41

if malla == 'C':
    points = airfoil_points // 3 * 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 2  # combadura
p = 4  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

# radio frontera externa
R = 20 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
# perfil.rotate(0)

archivo_perfil = 'perfil_final.csv'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

# mallaNACA.gen_Poisson()
direc = '/zero/'
mallaNACA = helpers.from_txt_mesh(filename='./potential_2412/' + direc
                                  + '/mallaNACA.txt_mesh')
mallaNACA.plot()

flag = 'r'
is_ok = False

while not is_ok:
    flag = input('Press \t[S] to save mesh,\n\t[N] to continue wihtout saving,\n\t'
             + '[n] to exit execution: ')
    print()
    if flag == 'S' or flag == 'N' or flag == 'n':
        is_ok = True

if flag == 'S':
    path = input('carpeta donde se va a guardar: ')
    try:
        mkdir(path)
    except:
        pass
elif flag == 'N':
    print('Continue without saving')
    pass
else:
    print('Quitting execution...')
    exit()

print('after mesh generation')
print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

# variables de flujo
t_inf = 293.15 # [K]
p_inf = 101325  # [Pa]
v_inf = 48 # [m / s]
# v_inf = 10 # [m / s]

###############################################################################
#
#   Con V = 48:
#           * restando p_inf a p0 dan cp de -72
#           * sin restar p_inf a p0 dan cp de 0 a 9
#   Con V = 10:
#           * restando p_inf a p0 dan cp de 0 a -0.07
#           * sin restar p_inf a p0 dan cp de 0 a 9
#
###############################################################################
alfa = 0

gamma = 1.4
cp_ = 1007
Rg = cp_ * (gamma - 1) / gamma
d_inf = p_inf / (Rg * t_inf)
h_inf = cp_ * t_inf
c_inf = (gamma * p_inf / d_inf) ** 0.5

h0 = h_inf + 0.5 * v_inf ** 2
d0 = d_inf / (1 - 0.5 * v_inf ** 2 / h0)
p0 = p_inf * (d0 / d_inf) ** gamma

mach_inf = v_inf / c_inf
Re = v_inf * c * d_inf / 18.25e-6

print('Re = ' + str(Re))
print('mach_inf = ' + str(mach_inf))
print('d_inf = ' + str(d_inf))
print('h0 = ' + str(h0))
print('d0 = ' + str(d0))
print('p0 = ' + str(p0))
if mach_inf > 0.8:
    print('Las condiciones de flujo son inválidas')
    exit()

(phi, C, theta, IMA) = potential_flow_o(d0, h0, gamma, mach_inf, v_inf,
                                            alfa, mallaNACA)
if flag == 'S':
    mallaNACA.to_txt_mesh(filename=(path + '/mallaNACA.txt_mesh'))
    np.savetxt(path + '/phi.csv', phi, delimiter=',')
    f = open(path + "/C.csv", "w+")
    f.write(str(C))
    f.close()
    np.savetxt(path + '/theta.csv', theta, delimiter=',')


(u, v) = velocity(alfa, C, mach_inf, theta, mallaNACA, phi, v_inf)
(cp, p) = pressure(u, v, v_inf, d_inf, gamma, p_inf, p0, d0, h0)
(psi, mach) = streamlines(u, v, gamma, h0, d0, p, mallaNACA)
(L, D) = lift_n_drag(mallaNACA, cp, 8, 1)

plt.figure('potential')
plt.contour(mallaNACA.X, mallaNACA.Y, phi, 95, cmap='jet')
plt.colorbar()
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
plt.axis('equal')
plt.colorbar()

plt.figure('pressure')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contourf(mallaNACA.X, mallaNACA.Y, cp, 105, cmap='jet')
plt.colorbar()
plt.axis('equal')

plt.figure('streamlines')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contour(mallaNACA.X, mallaNACA.Y, np.real(psi), 195, cmap='brg')
plt.colorbar()
plt.axis('equal')

plt.draw()
plt.show()
