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
from potential import potential_flow_o, velocity,\
                        pressure, streamlines, lift_n_drag
import util

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval,
                                                b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap
min_color = 0.25
max_color = 1.0
color_own = truncate_colormap(plt.get_cmap("inferno"), min_color, max_color)

file_name='/home/cardoso/Tesis/su2/jul_20/malla_C_multi_better/mesh_c_m.txt_mesh'
save_name='/home/cardoso/garbage/mesh_c_flap_su2.png'

mallaNACA = util.from_txt_mesh(filename=file_name)
# aspect = mallaNACA.get_aspect_ratio()
# skew = mallaNACA.get_skew()

fix, ax = plt.subplots(1)
ax.plot(mallaNACA.X, mallaNACA.Y, 'k', linewidth=0.8)
# ax.plot(mallaNACA.X[:, 100:], mallaNACA.Y[:, 100:], 'k', linewidth=1.3)
# ax.plot(mallaNACA.X[:, :100], mallaNACA.Y[:, :100], 'k', linewidth=0.3)
for i in range(mallaNACA.M):
    ax.plot(mallaNACA.X[i, :], mallaNACA.Y[i, :], 'b', linewidth=0.8)
ax.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k', linewidth=1.8)
ax.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k', linewidth=1.8)
ax.plot(mallaNACA.X[0, :], mallaNACA.Y[0, :], 'k', linewidth=1.8)
ax.plot(mallaNACA.X[-1, :], mallaNACA.Y[-1, :], 'k', linewidth=1.8)
# mesh_ = ax.pcolormesh(mallaNACA.X, mallaNACA.Y, aspect, cmap='viridis',
#                       rasterized=True, vmin=1.5, vmax=4.5)
# mesh_ = ax.pcolormesh(mallaNACA.X, mallaNACA.Y, skew, cmap=color_own,
#                       rasterized=True, vmin=0, vmax=1)
# plt.colorbar(mesh_, extend='both')

ax.set_title("Malla C - Análisis SU2")
# ax.set_xlim([-0.4, 1.1])
# ax.set_ylim([-0.6, 0.6])
ax.set_aspect('equal')
plt.savefig(save_name, bbox_inches='tight', pad_inches=0.05)
plt.show()

