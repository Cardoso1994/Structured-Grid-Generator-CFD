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
# variables de flujo
t_inf = 293.15
p_inf = 101325
v_inf = 48 # [m / s]
v_inf = 10 # [m / s]

alfa = 0

gamma = 1.4
cp_ = 1007

#           variables de flujo
# t_inf = 300
# p_inf = 101300
# v_inf = 10 # [m / s]
#
# alfa = 0
#
# gamma = 1.4
# cp_ = 1006
Rg = cp_ * (gamma - 1) / gamma
d_inf = p_inf / (Rg * t_inf)
h_inf = cp_ * t_inf
c_inf = (gamma * p_inf / d_inf) ** 0.5

h0 = h_inf + 0.5 * v_inf ** 2
d0 = d_inf / (1 - 0.5 * v_inf ** 2 / h0)
p0 = p_inf * (d0 / d_inf) ** gamma

mach_inf = v_inf / c_inf


path = '/home/desarrollo/'
direc = 'eight/'
mallaNACA = helpers.from_txt_mesh(filename='./potential_2412/' + direc
                                  + '/mallaNACA.txt_mesh')
phi = np.genfromtxt('./potential_2412/' + direc + '/phi.csv', delimiter=',')
# f = open("./potential_2412/five/C.csv", "r")
C = np.genfromtxt('./potential_2412/' + direc + '/C.csv', delimiter=',')
theta = np.genfromtxt('./potential_2412/' + direc + '/theta.csv',
                      delimiter=',')

# path = '/home/desarrollo/'
# mallaNACA = helpers.from_txt_mesh(filename='/home/desarrollo/garbage/eight/mallaNACA.txt_mesh')
# phi = np.genfromtxt('/home/desarrollo/garbage/eight/phi.csv', delimiter=',')
# # f = open("./potential_2412/five/C.csv", "r")
# C = np.genfromtxt('/home/desarrollo/garbage/eight/C.csv', delimiter=',')
# theta = np.genfromtxt('/home/desarrollo/garbage/eight/theta.csv', delimiter=',')

(u, v) = velocity(alfa, C, mach_inf, theta, mallaNACA, phi, v_inf)
(cp, p) = pressure(u, v, v_inf, d_inf, gamma, p_inf, p0, d0, h0)
(psi, mach) = streamlines(u, v, gamma, h0, d0, p, mallaNACA)
(L, D) = lift_n_drag(mallaNACA, cp, 8, 1)

plt.figure('potential')
plt.contour(mallaNACA.X, mallaNACA.Y, phi, 95, cmap='jet')
plt.colorbar()
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
plt.axis('equal')
plt.colorbar()

# plt.figure('velocity')
# plt.quiver(mallaNACA.X[:, 1:-1], mallaNACA.Y[:, 1:-1], u[:, 1:-1], v[:, 1:-1],
#            scale=90, scale_units='x')
# plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
# plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
# plt.axis('equal')

plt.figure('pressure')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contourf(mallaNACA.X, mallaNACA.Y, cp, 105, cmap='jet')
plt.colorbar()
plt.axis('equal')

plt.figure('streamlines')
plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
plt.contour(mallaNACA.X, mallaNACA.Y, np.real(psi), 195, cmap='brg')
plt.colorbar()
plt.axis('equal')

plt.draw()
plt.show()
