#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:53:21 2018

@author: Cardoso

Funciones de soporte
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import airfoil
import mesh
import mesh_o
import mesh_c


def from_txt_mesh(filename='./garbage/mesh_own.txt_mesh'):
    '''
    importa malla de archivo txt_mesh. formato propio
    '''

    if filename[-8:] != 'txt_mesh':
        print('WARNING!')
        print('La extensión del archivo a importar no coincide con la \
              extensión utilizada por este programa')
        print('FINALIZANDO EJECUCIÓN')

        exit()

        return

    with open(filename, 'r') as f:
        mesh = f.readlines()

    tipo = str(mesh[0].split()[-1])
    d_eta = float(mesh[1].split()[-1])
    d_xi = float(mesh[2].split()[-1])
    R = float(mesh[3].split()[-1])
    M = int(mesh[4].split()[-1])
    N = int(mesh[5].split()[-1])
    airfoil_alone = str(mesh[6].split()[-1])
    airfoil_join = int(mesh[7].split()[-1])

    if airfoil_alone == 'True':
        airfoil_alone = True
    elif airfoil_alone == 'False':
        airfoil_alone = False

    airfoil_boundary = np.fromstring(mesh[9], sep=',')
    X = np.zeros((M, N))
    Y = np.zeros((M, N))

    i = 0
    line = 11
    for line_ in range (line, M + line):
        X[i, :] = np.fromstring(mesh[line_], sep=',')
        i += 1

    line += M + 1
    i = 0
    for line_ in range (line, M + line):
        Y[i, :] = np.fromstring(mesh[line_], sep=',')
        i += 1

    perfil = airfoil.airfoil(c=1)
    perfil.x = X[:, 0]
    perfil.y = Y[:, 0]
    if tipo == 'O':
        # mesh = mesh_o.mesh_O(R, M, N, perfil)
        mesh = mesh_o.mesh_O(R, N, perfil)
    elif tipo == 'C':
        # mesh = mesh_c.mesh_C(R, M, N, perfil, from_file=True)
        mesh = mesh_c.mesh_C(R, N, perfil, from_file=True)

    # sea asignan atributos a malla
    mesh.d_eta = d_eta
    mesh.d_xi = d_xi
    mesh.R = R
    mesh.M = M
    mesh.N = N
    mesh.airfoil_alone = airfoil_alone
    mesh.airfoil_join = airfoil_join
    mesh.airfoil_boundary = airfoil_boundary
    mesh.X = X
    mesh.Y = Y

    return mesh

def get_size_airfoil(airfoil_boundary):
    '''
    Calcula el numero de puntos que forman un perfil basado en el array que
        define que puntos son parte de un perfil, de un flap o parte del
        dominio
    '''

    size_airfoil = 0

    for i in range(np.shape(airfoil_boundary)[0]):
        if airfoil_boundary[i] == 1:
            size_airfoil += 1

    return (size_airfoil)

def get_size_airfoil_n_flap(airfoil_boundary):
    '''
    Calcula el numero de puntos que forman un perfil con flap basado en el
        array que define que puntos son parte de un perfil, de un flap o parte
        del dominio
    '''
    size_airfoil    = 0
    size_flap       = 0

    for i in range(np.shape(airfoil_boundary)[0]):
        if airfoil_boundary[i] == 1:
            size_airfoil += 1
        elif airfoil_boundary[i] == 2:
            size_flap += 1

    return (size_airfoil, size_flap)

def get_aspect_ratio(mesh):
    '''
    Calcula el aspect ratio de cada celda. Para cualquier tipo de malla
    Basado en el método de:
        The Verdict Geometric Quality Library
    '''
    # X = np.zeros((mesh.M + 1, mesh.N + 1))
    # Y = np.zeros((mesh.M + 1, mesh.N + 1))
    aspect_ratio_ = np.zeros((mesh.M - 1, mesh.N - 1))

    # se calculan longitudes de los elementos de la celda
    for i in range(0, mesh.M - 1):
        for j in range(0, mesh.N -1):
            # dist = (l0, l1, l2, l3)
            dist = (((mesh.X[i, j] - mesh.X[i+1, j]) ** 2
                            + (mesh.Y[i, j] - mesh.Y[i+1, j]) ** 2) ** 0.5,
                    ((mesh.X[i+1, j] - mesh.X[i+1, j+1]) ** 2
                            + (mesh.Y[i+1, j] - mesh.Y[i+1, j+1]) ** 2) ** 0.5,
                    ((mesh.X[i+1, j+1] - mesh.X[i, j+1]) ** 2
                            + (mesh.Y[i+1, j+1] - mesh.Y[i, j+1]) ** 2) ** 0.5,
                    ((mesh.X[i, j+1] - mesh.X[i, j]) ** 2
                            + (mesh.Y[i, j+1] - mesh.Y[i, j]) ** 2) ** 0.5)

            max_dist = max(dist)

            dist_sum = dist[0] + dist[1] + dist[2] + dist[3]

            # cálculo de productos cruz
            l0_l1 = np.abs((mesh.X[i+1, j] - mesh.X[i, j])
                                *  (mesh.Y[i+1, j+1] - mesh.Y[i+1, j])
                            - (mesh.Y[i+1, j] - mesh.Y[i, j])
                                 *  (mesh.X[i+1, j+1] - mesh.X[i+1, j]))

            l2_l3 = np.abs((mesh.X[i, j+1] - mesh.X[i+1, j+1])
                                * (mesh.Y[i, j] - mesh.Y[i, j+1])
                            - (mesh.Y[i, j+1] - mesh.Y[i+1, j+1])
                                * (mesh.X[i, j] - mesh.X[i, j+1]))
            area = 0.5 * l0_l1 + 0.5 * l2_l3
            aspect_ratio_[i, j] = max_dist * dist_sum / 4 / area

    aspect_min = np.nanmin(aspect_ratio_)
    aspect_max = np.nanmax(aspect_ratio_)
    # cmap_ = cm.get_cmap('jet')

    print('aspect_max')
    print(aspect_max)
    print('aspect_min')
    print(aspect_min)
    plt.figure('aspect')
    plt.axis('equal')
    plt.plot(mesh.X, mesh.Y, 'k')
    plt.plot(mesh.X[:, 0], mesh.Y[:, 0], 'k')
    for i in range(mesh.M):
        plt.plot(mesh.X[i, :], mesh.Y[i, :], 'k')
    mesh_ = plt.pcolormesh(mesh.X, mesh.Y, aspect_ratio_, cmap='jet', rasterized=True,
                   vmin=(aspect_min),
                   vmax=(aspect_max))
    plt.colorbar(mesh_)

    # plt.pcolormesh(X, Y, aspect_ratio[1:, 1:])
    plt.draw()
    plt.show()

