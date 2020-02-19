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
# from analysis import potential_flow_o, potential_flow_o_esp
import helpers

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''

N = 65 # 37 # 60
union = 6

airfoil_points = 45 # 41

if malla == 'C':
    points = airfoil_points // 3 #* 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 4  # combadura
p = 4  # posicion de la combadura
t = 15  # espesor
c = 1  # cuerda [m]
# radio frontera externa
R = 20 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
# flap = airfoil.NACA4(m, p, t, 0.2 * c, number=2)
# flap.create_sin(points)
# flap.rotate(15)
# perfil.join(flap, dx=0.055, dy=0.05, union=union)
# perfil.rotate(3)

if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

# mallaNACA.gen_Laplace(metodo='SOR')
mallaNACA.gen_Poisson(metodo='SOR', omega=1.3, aa=20.5, cc=6.5, linea_eta=0)
print('after laplace')
print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

mallaNACA.plot()

is_ok = False

flag = 'r'

while not is_ok:
    flag = input('Press \t[S] to save mesh,\n\t[N] to continue without saving,'
             + '\n\t[n] to exit execution: ')
    if flag == 'S' or flag == 'N' or flag == 'n':
        is_ok = True
    print()

if flag == 'S':
    path = input('carpeta donde se va a guardar: ')
    try:
        mkdir(path)
        mallaNACA.to_txt_mesh(filename=(path + '/mallaNACA.txt_mesh'))
    except:
        pass
elif flag == 'N':
    print('Continue without saving')
else:
    print('Quitting execution...')
    exit()

aspect = helpers.get_aspect_ratio(mallaNACA)
skew = helpers.get_skew(mallaNACA)

aspect_min = np.nanmin(aspect)
aspect_max = np.nanmax(aspect)
skew_min = np.nanmin(skew)
skew_max = np.nanmax(skew)

aspect_min = 1.04805
aspect_max = 2.01482

limits = [[-20.5, 20.5, -20.5, 20.5], [-1., 1, -0.5, 0.5],
            [0.91, 1.01, -0.025, 0.025]]
for limit in limits:
    fig = plt.figure('malla_aspect')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([limit[0], limit[1]])
    ax.set_ylim([limit[2], limit[3]])
    ax.set_aspect('equal')
    # plt.axis('equal')
    ax.plot(mallaNACA.X, mallaNACA.Y, 'k', linewidth=0.5)
    ax.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k', linewidth=1.9)

    for i in range(mallaNACA.M):
        ax.plot(mallaNACA.X[i, :], mallaNACA.Y[i, :], 'k', linewidth=0.5)
    mesh_ = plt.pcolormesh(mallaNACA.X, mallaNACA.Y, aspect, cmap='jet', rasterized=True,
                   vmin=(aspect_min),
                   vmax=(aspect_max))
    plt.colorbar(mesh_, extend='both')

    fig = plt.figure('malla_skew')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([limit[0], limit[1]])
    ax.set_ylim([limit[2], limit[3]])
    ax.set_aspect('equal')
    # plt.axis('equal')
    ax.plot(mallaNACA.X, mallaNACA.Y, 'k', linewidth=0.5)
    ax.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k', linewidth=1.9)

    for i in range(mallaNACA.M):
        ax.plot(mallaNACA.X[i, :], mallaNACA.Y[i, :], 'k', linewidth=0.5)
    mesh_ = plt.pcolormesh(mallaNACA.X, mallaNACA.Y, skew, cmap='jet', rasterized=True,
                   vmin=(skew_min),
                   vmax=(skew_max))
    plt.colorbar(mesh_)

    plt.draw()
    plt.show()
