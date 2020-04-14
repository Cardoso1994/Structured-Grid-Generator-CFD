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
malla = 'C'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
N = 675
N = 275

union = 25

airfoil_points = 615

if malla == 'C':
    # points = airfoil_points // 3 * 2
    points = airfoil_points
elif malla == 'O':
    points = airfoil_points

print('airofil points: ' + str(points))
# datos de perfil NACA
m = 0  # combadura
p = 0  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]
# radio frontera externa
R = 50 * c

perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(points)
flap = airfoil.NACA4(m, p, t, 0.2 * c, number=2)
flap.create_sin(points)
flap.rotate(15)
perfil.join(flap, dx=0.055, dy=0.01, union=union)
# perfil.rotate(30)

M = np.shape(perfil.x)[0]

archivo_perfil = 'perfil_final.csv'
if malla == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil, weight=1.13)
    # mallaNACA = mesh_c.mesh_C(R, M, N, perfil.x, perfil.y, False, perfil.union,
    #                           perfil.is_boundary, weight=1.129)


# normal
# mallaNACA.gen_Poisson_v(metodo='SOR', omega=0.5, aa=69.95, cc=7.7, linea_eta=0)
# mallaNACA.gen_Poisson_v(metodo='SOR', omega=0.5, aa=50500, cc=3, linea_eta=0)

# sectioned in 4
# mallaNACA.gen_Poisson_v_4(metodo='SOR', omega=0.3, aa=139.95, cc=7.7, linea_eta=0)
# mallaNACA.gen_Poisson_v_4(metodo='SOR', omega=0.3, aa=0.95, cc=7.7, linea_eta=0)
# mallaNACA.gen_Poisson_v_(metodo='SOR', omega=0.3, aa=159.95, cc=0.2, linea_eta=0)
# mallaNACA.gen_Poisson_v_(metodo='SOR', omega=0.5, aa=60500,
#                                  cc=7, linea_eta=0)
mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.7, aa=65, cc=7, linea_eta=0)



mallaNACA.to_su2('/home/desarrollo/garbage/mesh_c_flap.su2')
mallaNACA.to_txt_mesh('/home/desarrollo/garbage/mesh_c_flap.txt_mesh')

print('Malla generada')
mallaNACA.plot()

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
