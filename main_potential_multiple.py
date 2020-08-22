#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:33:47 2018

@author: cardoso
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser

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
N_1 = 22
N = 80
N1 = 189
split = 8

home = expanduser('~')

# points = 11
airfoil_points = 507

weight = 1.15
a = 0.01
a = 0
c = 8.7
c = 0
linea_xi = 0.5
linea_xi = 0

points = airfoil_points

# datos de perfil NACA
m = 0  # combadura
p = 0  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

# radio frontera externa
R = 210 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)

M = np.shape(perfil.x)[0]
print(f"shape perfil: {M}")

mallaNACA_1 = mesh_o.mesh_O(R, N_1, perfil)
mallaNACA_1.gen_TFI()

mallaNACA = mesh_o.mesh_O(R, N, perfil)
mallaNACA.X[:, -1] = mallaNACA_1.X[:, -20]
mallaNACA.Y[:, -1] = mallaNACA_1.Y[:, -20]
mallaNACA1 = mesh_o.mesh_O(R, N1, perfil)

print(f"shape mesh: {np.shape(mallaNACA.X)[0]}")
# print('M = ' + str(mallaNACA.M))
# print('N = ' + str(mallaNACA.N))

mallaNACA = util.from_txt_mesh(
        filename=(home + '/garbage/mesh_o.txt_mesh'))
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.15, a=a, c=c, linea_xi=linea_xi,
#                         aa=58.5, cc=8.4, linea_eta=0)
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.15, a=a, c=c, linea_xi=linea_xi,
#                         aa=108.5, cc=13.4, linea_eta=0)
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.15, a=a, c=c, linea_xi=linea_xi,
#                         aa=210.1, cc=8.5, linea_eta=0)

# mallaNACA.to_su2(home + '/garbage/mesh_o.su2')

mallaNACA.to_txt_mesh(home + '/garbage/mesh_o.txt_mesh')

plt.figure('MALLA NACA')
plt.title('MALLA NACA')
mallaNACA.plot()

# malla de 50 a 70 m
mallaNACA_1.X[:, 0] = mallaNACA.X[:, -1]
mallaNACA_1.Y[:, 0] = mallaNACA.Y[:, -1]
mallaNACA_1.gen_Poisson_n(metodo='SOR', omega=0.15, a=a, c=c,
                          linea_xi=linea_xi, aa=0.0035, cc=6.8, linea_eta=0)

plt.figure('MALLA NACA 1')
plt.title('MALLA NACA 1')
mallaNACA_1.plot()

# malla de las primeras lineas de la malla
mallaNACA1.X[:, -1] = mallaNACA.X[:, split]
mallaNACA1.Y[:, -1] = mallaNACA.Y[:, split]

mallaNACA1.gen_Poisson_n(metodo='SOR', omega=0.05, a=a, c=c,
                         linea_xi=linea_xi, aa= 1597000,
                         cc=4.9, linea_eta=0)
# mallaNACA1.gen_Poisson_n(metodo='SOR', omega=0.05, a=a, c=c,
#                          linea_xi=linea_xi, aa= 200 * 19597000,
#                          cc=23.0, linea_eta=0)

plt.figure('MALLA NACA_1')
plt.title('MALLA NACA_1')
mallaNACA1.plot()

mallaNACA_2 = mesh_o.mesh_O(1, N + N1 - 1 - split + N_1 - 1, perfil)

mallaNACA_2.X[:, :N + N1 - split - 1] = np.concatenate((mallaNACA1.X[:, :],
                                      mallaNACA.X[:, split+1:]), axis=1)
mallaNACA_2.Y[:, :N + N1 - split - 1] = np.concatenate((mallaNACA1.Y[:, :],
                                      mallaNACA.Y[:, split+1:]), axis=1)

mallaNACA_2.X[:, N + N1 - split - 1:] = mallaNACA_1.X[:, 1:]
mallaNACA_2.Y[:, N + N1 - split - 1:] = mallaNACA_1.Y[:, 1:]
print(f"mallaNACA_2 M: {np.shape(mallaNACA_2.X)[0]} " +
      f"N: {np.shape(mallaNACA_2.X)[1]}")
plt.figure('MALLA NACA 2')
plt.title('MALLA NACA 2')
mallaNACA_2.plot()

mallaNACA_2.to_txt_mesh(home + '/garbage/mesh_o_m.txt_mesh')

