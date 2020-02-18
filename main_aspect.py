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

N = 35 # 37 # 60
union = 6

airfoil_points = 77 # 41

if malla == 'C':
    points = airfoil_points // 3 #* 2
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
mallaNACA.gen_Poisson(metodo='SOR')
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

helpers.get_aspect_ratio(mallaNACA)
