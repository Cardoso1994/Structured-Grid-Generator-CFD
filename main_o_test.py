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
from potential import potential_flow_o
import util

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
N = 35

union = 25
airfoil_points = 175

if malla == 'C':
    points = airfoil_points // 3  # * 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 2  # combadura
p = 44 # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

# radio frontera externa
R = 40 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
flap = airfoil.NACA4(m, p, t, 0.2 * c, number=2)
flap.create_sin(points)
flap.rotate(15)
perfil.join(flap, dx=0.055, dy=0.01, union=union)

archivo_perfil = 'perfil_final.csv'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil, weight=1.15)

print('')
print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

mallaNACA.gen_inter_pol(eje='eta')
mallaNACA.plot()
mallaNACA.get_aspect_ratio()
mallaNACA.get_skew()
exit()
# mallaNACA.gen_Laplace(metodo='SOR', omega=1.2)
# mallaNACA.gen_Laplace_v_(metodo='SOR', omega=0.7)
# mallaNACA.gen_Laplace_n(metodo='SOR', omega=1.3)
# mallaNACA.gen_Poisson(metodo='SOR', omega=0.7, aa=45, cc=17.4, linea_eta=0)
# mallaNACA.gen_Poisson_v_(metodo='SOR', omega=0.7, aa=12.5, cc=5, linea_eta=0)
mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.3, aa=68, cc=17, linea_eta=0)


mallaNACA.to_su2('/home/desarrollo/garbage/__mesh_o_____.su2')
mallaNACA.to_txt_mesh('/home/desarrollo/garbage/mesh_o_____.txt_mesh')

file_name = '/home/desarrollo/garbage/mesh_o_____.txt_mesh'
mallaNACA__ = util.from_txt_mesh(filename=file_name)

print("after importing")
mallaNACA__.plot()
print("after ploting")
