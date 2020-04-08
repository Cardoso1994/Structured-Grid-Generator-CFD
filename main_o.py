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
N = 335
union = 49

# points = 11
airfoil_points = 335 # 499
airfoil_points = 859
airfoil_points = 259
airfoil_points = 749

if malla == 'C':
    points = airfoil_points // 3  # * 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 0  # combadura
p = 0  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

# radio frontera externa
R = 40 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)

archivo_perfil = 'perfil_final.csv'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

# mallaNACA.gen_Poisson(metodo='SOR', omega=0.7, aa=245, cc=7.4, linea_eta=0)
# mallaNACA.gen_Poisson(metodo='SOR', omega=0.7, aa=145, cc=3.7, linea_eta=0)
# mallaNACA.gen_Poisson_v(metodo='SOR', omega=0.7, aa=12.5, cc=5, linea_eta=0)
mallaNACA.gen_Poisson_v(metodo='SOR', omega=0.7, aa=230.5, cc=7.4, linea_eta=0)

mallaNACA.to_su2('/home/desarrollo/garbage/mesh_o.su2')
mallaNACA.to_txt_mesh('/home/desarrollo/garbage/mesh_o.txt_mesh')

mallaNACA.plot()
