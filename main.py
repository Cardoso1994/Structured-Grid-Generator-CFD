#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
# import helpers
import util

# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
N = 75
# N = 300

union = 39

# points = 11
airfoil_points = 399 # 499
airfoil_points = 549

airfoil_points = 89 # 499
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
R = 50 * c

perfil = airfoil.NACA4(m, p, t, c)
# perfil = airfoil.NACA4(m, p, t, 0.25 * c)
perfil.create_sin(points)
# perfil.rotate(30)
# perfil_1 = airfoil.NACA4(m, p, t, 0.25 * c, number=2)
# perfil_2 = airfoil.NACA4(m, p, t, 0.25 * c, number=3)
# perfil_1.create_sin(points)
# perfil_2.create_sin(points)
# perfil.join(perfil_1, dx=0.05)
# perfil.join(perfil_2, dx=0.05)
flap = airfoil.NACA4(m, p, t, 0.2 * c, number=2)
flap.create_sin(points - 25)
flap.rotate(15)
perfil.join(flap, dx=0.05, dy=0.05)

# perfil = airfoil.airfoil(1)
# perfil.create("./whitcomb-il.txt")
# perfil.plot("Whitcomb-il", save='/home/cardoso/garbage/whitcomb.png')


M = np.shape(perfil.x)[0]

archivo_perfil = 'perfil_final.csv'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.75, aa=500, cc=8, linea_eta=0)
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.15, aa=1500, cc=12, linea_eta=0)
# mallaNACA.gen_Poisson(metodo='SOR', omega=0.7, aa=185, cc=3.7, linea_eta=0)
# mallaNACA.gen_Poisson_v_(metodo='SOR', omega=0.5, aa=650,
#                                  cc=7, linea_eta=0)
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.15, aa=1500, cc=12, linea_eta=0)
# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.3, aa=21550, cc=21, linea_eta=0)

title = "Malla O"
file_save = "/home/cardoso/Tesis/presentacion/img/malla_o.png"
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.set_xlim([-0.3, 0.75])
# ax.set_ylim([-0.1, 0.6])
ax.set_aspect('equal')
ax.set_title(title)
ax.plot(mallaNACA.X, mallaNACA.Y, 'b')
for i in range(mallaNACA.M):
    ax.plot(mallaNACA.X[i, :], mallaNACA.Y[i, :], 'b')

plt.savefig(file_save, bbox_inches='tight', pad_inches=0.05)
plt.show()
mallaNACA.to_su2('/home/cardoso/garbage/mesh.su2')
mallaNACA.to_txt_mesh('/home/cardoso/garbage/mesh.txt_mesh')

plt.plot()
exit(12)
mallaNACA.plot()
print('after mesh generation')
print('M = ' + str(mallaNACA.M))
print('N = ' + str(mallaNACA.N))

flag = 'r'
is_ok = False

while not is_ok:
    flag = input('Press \t[S] to save mesh,\n\t[N] to continue wihtout'  +
                 'saving,\n\t [n] to exit execution: ')
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
