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
N = 675
N1 = 219

# points = 11
airfoil_points = 617
airfoil_points = 289


weight = 1.3
a = 0.01
a = 0
c = 8.7
c = 0
linea_xi = 0.5
linea_xi = 0

split = 70

if malla == 'C':
    points = airfoil_points
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 0  # combadura
p = 0  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

# radio frontera externa
R = 70 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)

M = np.shape(perfil.x)[0]
print(f"shape perfil: {M}")


mallaNACA = mesh_o.mesh_O(R, N, perfil)
mallaNACA_1 = mesh_o.mesh_O(R, N1, perfil)

print(f"shape mesh: {np.shape(mallaNACA.X)[0]}")
# print('M = ' + str(mallaNACA.M))
# print('N = ' + str(mallaNACA.N))

mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.15, a=a, c=c, linea_xi=linea_xi,
                        aa=156.5, cc=8.5, linea_eta=0)
# mallaNACA = util.from_txt_mesh(
#         filename='/home/desarrollo/garbage/mesh_o.txt_mesh')

mallaNACA.to_su2('/home/desarrollo/garbage/mesh_o.su2')
mallaNACA.to_txt_mesh('/home/desarrollo/garbage/mesh_o.txt_mesh')

plt.figure('MALLA NACA')
plt.title('MALLA NACA')
mallaNACA.plot()

mallaNACA_1.X[:, -1] = mallaNACA.X[:, split]
mallaNACA_1.Y[:, -1] = mallaNACA.Y[:, split]

mallaNACA_1.gen_Poisson_n(metodo='SOR', omega=0.20, a=a, c=c,
                          linea_xi=linea_xi, aa=55000, cc=2.5, linea_eta=0)

plt.figure('MALLA NACA 1')
plt.title('MALLA NACA 1')
mallaNACA_1.plot()

mallaNACA_2 = mesh_o.mesh_O(1, N + N1 - 1 - split, perfil)

mallaNACA_2.X[:, :] = np.concatenate((mallaNACA_1.X[:, :],
                                      mallaNACA.X[:, split+1:]), axis=1)
mallaNACA_2.Y[:, :] = np.concatenate((mallaNACA_1.Y[:, :],
                                      mallaNACA.Y[:, split+1:]), axis=1)

print(f"mallaNACA_2 M: {np.shape(mallaNACA_2.X)[0]} " +
      f"N: {np.shape(mallaNACA_2.X)[1]}")
plt.figure('MALLA NACA 2')
plt.title('MALLA NACA 2')
mallaNACA_2.plot()

mallaNACA_2.to_su2('/home/desarrollo/garbage/mesh_o_m.su2')
mallaNACA_2.to_txt_mesh('/home/desarrollo/garbage/mesh_o_m.txt_mesh')

exit()

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

mallaNACA.to_txt_mesh(path + '/mallaNACA.txt_mesh')

exit()



# variables de flujo
t_inf = 273.15
p_inf = 101325
v_inf = 75

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
print(mach_inf)
print(Re)
(phi, C, theta, IMA) = potential_flow_o_esp(d0, h0, gamma, mach_inf, v_inf, alfa, mallaNACA)

plt.figure('potential')
plt.plot(X[:, N-1], Y[:, N-1], 'k')
plt.contour(X, Y, phi)
