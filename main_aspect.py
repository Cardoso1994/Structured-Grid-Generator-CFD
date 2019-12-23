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
from analysis import potential_flow_o, potential_flow_o_esp
import helpers

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
N = 13
union = 6

# points = 11
airfoil_points = 35

if malla == 'C':
    points = airfoil_points // 3 * 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 1  # combadura
p = 2  # posicion de la combadura
t = 10  # espesor
c = 1  # cuerda [m]
# radio frontera externa
R = 20 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
# flap = airfoil.NACA4(m, p, t, 0.2 * c, number=2)
# flap.create_sin(points)
# flap.rotate(10)
# perfil.join(flap, dx=0.055, dy=0.05, union=union)
# perfil.rotate(30)
M = np.shape(perfil.x)[0]

archivo_perfil = './garbage/perfil_final.csv'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

perfil.to_csv(archivo_perfil)
# mallaNACA.gen_Poisson(metodo='SOR')
mallaNACA.gen_Laplace(metodo='GS')
# mallaNACA.gen_TFI()
print('after laplace')
print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

helpers.get_aspect_ratio(mallaNACA)
