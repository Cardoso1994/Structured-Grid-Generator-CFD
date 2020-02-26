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
from potential import potential_flow_o, potential_flow_o_esp, velocity,\
                        pressure, streamlines, lift_n_drag
import helpers

# datos de perfil NACA
m = 2  # combadura
p_ = 4  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

# radio frontera externa
R = 20 * c

# variables de flujo
t_inf = 293.15 # [K]
p_inf = 101325  # [Pa]
v_inf = 48 # [m / s]

alfa = 0

gamma = 1.4
cp_ = 1007
mu = 18.25e-6
Rg = cp_ * (gamma - 1) / gamma
d_inf = p_inf / (Rg * t_inf)
h_inf = cp_ * t_inf
c_inf = (gamma * p_inf / d_inf) ** 0.5

# relaciones isentrópicas
h0 = h_inf + 0.5 * v_inf ** 2
# de cengel termo
d0 = d_inf * (1 + (gamma - 1) / 2 * (v_inf / c_inf) ** 2) ** (1 / (gamma - 1))
# d0 = d_inf / (1 - 0.5 * v_inf ** 2 / h0)
p0 = p_inf * (d0 / d_inf) ** gamma

mach_inf = v_inf / c_inf
Re = v_inf * d_inf / mu


path = '/home/cardoso/'
direc = 'ten'
mallaNACA = helpers.from_txt_mesh(filename='./potential_2412/' + direc
                                  + '/mallaNACA.txt_mesh')
phi = np.genfromtxt('./potential_2412/' + direc + '/phi.csv', delimiter=',')
# f = open("./potential_2412/five/C.csv", "r")
C = np.genfromtxt('./potential_2412/' + direc + '/C.csv', delimiter=',')
theta = np.genfromtxt('./potential_2412/' + direc + '/theta.csv',
                      delimiter=',')

(u, v) = velocity(alfa, C, mach_inf, theta, mallaNACA, phi, v_inf)
(cp, p) = pressure(u, v, v_inf, d_inf, gamma, p_inf, p0, d0, h0)
(psi, mach) = streamlines(u, v, gamma, h0, d0, p, mallaNACA)
(L, D) = lift_n_drag(mallaNACA, cp, 10, 1)
print("L = " + str(L))
print("D = " + str(D))

points = mallaNACA.M
print(points)
perfil = airfoil.NACA4(m, p_, t, c)
perfil.create_sin(points)
perfil.x += 0.25
x_perf = perfil.x
y_perf = perfil.y
cp_perf =cp[:, 0]

plt.figure('perf')
plt.plot(x_perf, y_perf * 7, 'b')
plt.plot(x_perf[1:points // 2 + 2], cp_perf[1:points // 2 + 2], 'r', label='intrados')
plt.plot(x_perf[points // 2 + 1:-1], cp_perf[points // 2 + 1:-1], 'k', label='extrados')
plt.legend(loc='upper right')
# plt.axis('equal')
plt.draw()
plt.show()

mallaNACA.plot()

map_ = 'viridis'
limits = [[-20.5, 20.5, -20.5, 20.5], [-1.25, 1.25, -0.8, 0.8]]
for limit in limits:
    fig = plt.figure('malla_aspect')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([limit[0], limit[1]])
    ax.set_ylim([limit[2], limit[3]])
    ax.set_aspect('equal')
    # plt.axis('equal')
    # ax.plot(mallaNACA.X, mallaNACA.Y, 'k', linewidth=0.5)
    ax.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k', linewidth=1.9)
    ax.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k', linewidth=1.9)

    # for i in range(mallaNACA.M):
    #     ax.plot(mallaNACA.X[i, :], mallaNACA.Y[i, :], 'k', linewidth=0.5)
    mesh_ = plt.contour(mallaNACA.X, mallaNACA.Y, psi, 495,
                           cmap=map_)
    #plt.colorbar(mesh_)
    plt.show()
    plt.draw()

plt.figure('potential')
plt.contour(mallaNACA.X, mallaNACA.Y, phi, 95, cmap=map_)
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
plt.axis('equal')

plt.figure('pressure')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contourf(mallaNACA.X, mallaNACA.Y, cp, 75, cmap=map_)
plt.colorbar()
plt.axis('equal')

plt.figure('streamlines')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contour(mallaNACA.X, mallaNACA.Y, np.real(psi), 195, cmap=map_)
plt.axis('equal')

plt.draw()
plt.show()
