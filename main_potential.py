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
from potential import potential_flow_o, velocity, pressure, lift_n_drag,\
    streamlines, potential_flow_o_n
# from potential_performance import potential_flow_o_n
# import helpers
import util

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''

N = 339
N = 649
airfoil_points = 619
airfoil_points = 229

if malla == 'C':
    points = airfoil_points // 3 * 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 2  # combadura
p = 4  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

p_ = p

# radio frontera externa
R = 55 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
# perfil.rotate(2)

archivo_perfil = 'perfil_final.csv'

mallaNACA = mesh_o.mesh_O(R, N, perfil)

# mallaNACA.gen_Poisson(omega=1.3, aa=40, cc=6.8, linea_eta=0)
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=1.3, aa=40, cc=6.8, linea_eta=0)
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.5, aa=320, cc=8.9, linea_eta=0)
# direc = '/four-/'
mallaNACA = util.from_txt_mesh(
        filename='./potential_2412_mayo/mallaNACA_8.txt_mesh')

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
        mallaNACA.to_txt_mesh(filename=(path + '/mallaNACA' + '.txt_mesh'))
    except:
        mallaNACA.to_txt_mesh(filename=(path + '/mallaNACA' + '.txt_mesh'))
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

alfa = 5

gamma = 1.4
cp_ = 1007
mu = 18.25e-6
Rg = cp_ * (gamma - 1) / gamma
d_inf = p_inf / (Rg * t_inf)
h_inf = cp_ * t_inf
c_inf = (gamma * p_inf / d_inf) ** 0.5

# relaciones isentrópicas
h0 = h_inf + 0.5 * v_inf ** 2
# de cengel termo
d0 = d_inf * (1 + (gamma - 1) / 2 * (v_inf / c_inf) ** 2) ** (1 / (gamma - 1))
# d0 = d_inf / (1 - 0.5 * v_inf ** 2 / h0)
p0 = p_inf * (d0 / d_inf) ** gamma

mach_inf = v_inf / c_inf
Re = v_inf * c * d_inf / mu

print('Re = ' + str(Re))
print('mach_inf = ' + str(mach_inf))
print('d_inf = ' + str(d_inf))
print('h0 = ' + str(h0))
print('d0 = ' + str(d0))
print('p0 = ' + str(p0))

if mach_inf > 0.8:
    print('Las condiciones de flujo son inválidas')
    exit()

alfas = ['-4', '-2', '0', '2', '4', '6', '8', '10']
alfas = ['-4', '-2', '0', '2', '4', '6']
# alfas = ['8', '10']

for alfa_ in alfas:
    alfa = int(alfa_)
    # (phi, C, theta, IMA) = potential_flow_o(d0, h0, gamma, mach_inf, v_inf,
    #                                          alfa, mallaNACA)
    (phi, C, theta, IMA) = potential_flow_o_n(d0, h0, gamma, mach_inf, v_inf,
                                             alfa, mallaNACA)

    if flag == 'S':
        mallaNACA.to_txt_mesh(filename=(path + '/mallaNACA_' + str(alfa)
                                        + '.txt_mesh'))
        np.savetxt(path + '/phi_' + str(alfa) + '.csv', phi, delimiter=',')
        f = open(path + '/C_' + str(alfa) + '.csv', "w+")
        f.write(str(C))
        f.close()
        np.savetxt(path + '/theta_' + str(alfa) + '.csv', theta, delimiter=',')

    (u, v) = velocity(alfa, C, mach_inf, theta, mallaNACA, phi, v_inf)
    (cp, p) = pressure(u, v, v_inf, d_inf, gamma, p_inf, p0, d0, h0)
    (psi, mach) = streamlines(u, v, gamma, h0, d0, p, mallaNACA)
    (L, D) = lift_n_drag(mallaNACA, cp, alfa, c)

    print(f"perfil NACA {m}{p_}{t}")
    print(f"alfa = {alfa_}")
    print(f"L = {L}")
    print(f"D = {D}")

plt.figure('potential')
plt.contour(mallaNACA.X, mallaNACA.Y, phi, 95, cmap='jet')
plt.colorbar()
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
plt.axis('equal')

plt.figure('pressure')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contourf(mallaNACA.X, mallaNACA.Y, cp, 75, cmap='jet')
plt.colorbar()
plt.axis('equal')

plt.figure('_pressure')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contourf(mallaNACA.X, mallaNACA.Y, p - p_inf, 75, cmap='jet')
plt.colorbar()
plt.axis('equal')

plt.figure('pressure_')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contour(mallaNACA.X, mallaNACA.Y, cp, 75, cmap='jet')
plt.colorbar()
plt.axis('equal')

plt.figure('streamlines')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contour(mallaNACA.X, mallaNACA.Y, np.real(psi), 195, cmap='brg')
plt.colorbar()
plt.axis('equal')

plt.draw()
plt.show()
