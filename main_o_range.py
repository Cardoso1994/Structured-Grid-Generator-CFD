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
N = 45 # 60
union = 6

airfoil_points = 35 # 45

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
flap = airfoil.NACA4(m, p, t, 0.2 * c, number=2)
flap.create_sin(points)
flap.rotate(40)
perfil.join(flap, dx=0.055, dy=0.05, union=union)
# perfil.rotate(-90)

if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

# mallaNACA.gen_Laplace(metodo='SOR')
mallaNACA.gen_Poisson(metodo='SOR')
print('after laplace')
# mallaNACA.plot()
limits = [[-0.4, 1.2, -0.3, 0.5], [0.69, 0.85, 0.12, 0.20],
            [0.91, 1.01, -0.025, 0.025]]
for limit in limits:
    fig = plt.figure('malla')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([limit[0], limit[1]])
    ax.set_ylim([limit[2], limit[3]])
    ax.set_aspect('equal')
    # plt.axis('equal')
    ax.plot(mallaNACA.X, mallaNACA.Y, 'k', linewidth=1.5)
    ax.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k', linewidth=1.9)

    for i in range(mallaNACA.M):
        ax.plot(mallaNACA.X[i, :], mallaNACA.Y[i, :], 'b', linewidth=1.5)
    plt.draw()
    plt.show()

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

helpers.get_aspect_ratio(mallaNACA)
