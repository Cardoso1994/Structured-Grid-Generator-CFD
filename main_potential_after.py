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
m = 0  # combadura
p_ = 0  # posicion de la combadura
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

path = './potential_final_2412_2/'
# alfas = ['-4', '-2', '0', '2', '4', '6', '8', '10']
alfas = ['-4', '0', '4', '8']
alfas_gr = list(map(int, alfas))
cl_0012 = [-0.4543, -0.2218, 0.001, 0.2239, 0.4566, 0.6928, 0.9306, 1.1713 ]
cl_0012_abbott = [-0.44, -0.22, 0, 0.22, 0.44, 0.66, 0.88, 1.1 ]
# cl_2412 = [-0.2238, -0.0217, 0.2673, 0.5126, 0.7574, 1.0014, 1.2445, 1.4863 ]
cl_2412 = [-0.2138, -0.0017, 0.2573, 0.4926, 0.7174, 0.9414, 1.1445, 1.3463 ]
cl_2412_abbott = [-0.19, 0.04, 0.25, 0.47, 0.67, 0.89, 1.09, 1.26 ]


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(alfas_gr, cl_0012, 'k', label='flujo potencial')
# ax.plot(alfas_gr, cl_0012, '*k')# , label='flujo potencial')
# ax.plot(alfas_gr, cl_0012_abbott, 'r', label='Abbott')
# ax.plot(alfas_gr, cl_0012_abbott, '*r')# , '*r', label='Abbott')
# plt.xlabel('alfa')
# plt.ylabel('cl')
# plt.legend(loc='upper left')
# ax.set_aspect(15)
# ax.grid(True)
# plt.show()

map_ = 'viridis'
cps = np.zeros((149, 4))
j = 0
for alfa in alfas:
    mallaNACA = helpers.from_txt_mesh(filename=path
                                  + '/mallaNACA_' + alfa +'.txt_mesh')
    phi = np.genfromtxt(path + '/phi_' + alfa + '.csv', delimiter=',')
    C = np.genfromtxt(path + '/C_' + alfa + '.csv', delimiter=',')
    theta = np.genfromtxt(path + '/theta_' + alfa + '.csv',
                      delimiter=',')

    (u, v) = velocity(int(alfa), C, mach_inf, theta, mallaNACA, phi, v_inf)
    (cp, p) = pressure(u, v, v_inf, d_inf, gamma, p_inf, p0, d0, h0)
    (psi, mach) = streamlines(u, v, gamma, h0, d0, p, mallaNACA)
    (L, D) = lift_n_drag(mallaNACA, cp, int(alfa), 1)
    cps[:, j] = cp[:, 0]
    j += 1
    print('alfa = ' + alfa)
    print("L = " + str(L))
    print("D = " + str(D))

    # plt.figure('potential')
    # plt.contour(mallaNACA.X, mallaNACA.Y, phi, 95, cmap=map_)
    # plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
    # plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
    # plt.axis('equal')

    # plt.figure('pressure')
    # plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
    # plt.contourf(mallaNACA.X, mallaNACA.Y, cp, 180, cmap=map_)
    # plt.colorbar()
    # plt.axis('equal')

    # plt.figure('streamlines')
    # plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
    # plt.contour(mallaNACA.X, mallaNACA.Y, np.real(psi), 195, cmap=map_)
    # plt.axis('equal')

    # plt.draw()
    # plt.show()
    # limits = [[-20.5, 20.5, -20.5, 20.5], [-1.25, 1.25, -0.8, 0.8]]
    # for limit in limits:
    #     fig = plt.figure('malla_aspect')
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.set_xlim([limit[0], limit[1]])
    #     ax.set_ylim([limit[2], limit[3]])
    #     ax.set_aspect('equal')
    #     ax.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k', linewidth=1.9)
    #     ax.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k', linewidth=1.9)

    #     mesh_ = plt.contour(mallaNACA.X, mallaNACA.Y, psi, 295, #185
    #                         cmap=map_)
    #     # plt.colorbar(mesh_)
    #     plt.show()
    #     plt.draw()




# path = '/home/cardoso/'
# # direcs = ['four-', 'two-', 'zero', 'two', 'four', 'six', 'eight', 'ten']
# direcs = ['-4', '0', '4', '8']
# for direc in direcs:
#     mallaNACA = helpers.from_txt_mesh(filename=path + direc
#                                       + '/mallaNACA.txt_mesh')
#     phi = np.genfromtxt('./potential_2412/' + direc + '/phi.csv', delimiter=',')
#     # f = open("./potential_2412/five/C.csv", "r")
#     C = np.genfromtxt('./potential_2412/' + direc + '/C.csv', delimiter=',')
#     theta = np.genfromtxt('./potential_2412/' + direc + '/theta.csv',
#                           delimiter=',')
#
#     (u, v) = velocity(alfa, C, mach_inf, theta, mallaNACA, phi, v_inf)
#     (cp, p) = pressure(u, v, v_inf, d_inf, gamma, p_inf, p0, d0, h0)
#     (psi, mach) = streamlines(u, v, gamma, h0, d0, p, mallaNACA)
#     (L, D) = lift_n_drag(mallaNACA, cp, 10, 1)
#     cps[:, j] = cp[:, 0]
#     j += 1
#     print("L = " + str(L))
#     print("D = " + str(D))


points = mallaNACA.M
perfil = airfoil.NACA4(m, p_, t, c)
perfil = airfoil.NACA4(2, 4, 12, 1)
perfil.create_sin(points)
perfil.x += 0.25
x_perf = perfil.x
y_perf = perfil.y * -1
cp_perf =cp[:, 0]

plt.figure('perf')
plt.plot(x_perf, y_perf * 8.3, 'k')

for j in range(4):
    plt.plot(x_perf[1:-1], cps[1:-1, j], label=(alfas[j]))
    # plt.plot(x_perf[1: points // 2 + 1], cps[1: points // 2 + 1, j], label=(direcs[j]))
    # plt.plot(x_perf[points // 2 + 1 :], cps[points // 2 + 1 :, j], label=(direcs[j]))
    # plt.plot(x_perf[points // 2 + 1:-1], cp_perf[points // 2 + 1:-1], 'k', label='extrados')
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
# plt.axis('equal')
plt.draw()
plt.show()
exit()

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
    # ax.plot(mallaNACA.X, mallaNACA.Y, 'b', linewidth=1.5)
    ax.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k', linewidth=1.9)
    ax.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k', linewidth=1.9)

    # for i in range(mallaNACA.M):
    #     ax.plot(mallaNACA.X[i, :], mallaNACA.Y[i, :], 'b', linewidth=1.5)
    mesh_ = plt.contour(mallaNACA.X, mallaNACA.Y, phi, 195,
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
