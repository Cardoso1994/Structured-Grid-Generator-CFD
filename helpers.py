#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:53:21 2018

@author: Cardoso

Funciones de soporte
"""

import numpy as np
import matplotlib.pyplot as plt

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
    '''
    X = np.zeros((mesh.M + 1, mesh.N + 1))
    Y = np.zeros((mesh.M + 1, mesh.N + 1))

    X[0, 0] = mesh.X[0, 0]
    X[0, -1] = mesh.X[0, -1]
    X[-1, 0] = mesh.X[-1, 0]
    X[-1, -1] = mesh.X[-1, -1]

    Y[0, 0] = mesh.Y[0, 0]
    Y[0, -1] = mesh.Y[0, -1]
    Y[-1, 0] = mesh.Y[-1, 0]
    Y[-1, -1] = mesh.Y[-1, -1]

    for j in range(1, mesh.N):
        X[0, j] = (mesh.X[0, j] + mesh.X[0, j-1]) / 2
        Y[0, j] = (mesh.Y[0, j] + mesh.Y[0, j-1]) / 2
    X[-1, 1:-1] = X[0, 1:-1]
    Y[-1, 1:-1] = Y[0, 1:-1]

    for i in range(1, mesh.M):
        X[i, 0] = (mesh.X[i, 0] + mesh.X[i-1, 0]) / 2
        Y[i, 0] = (mesh.Y[i, 0] + mesh.Y[i-1, 0]) / 2
        X[i, -1] = (mesh.X[i, -1] + mesh.X[i-1, -1]) / 2
        Y[i, -1] = (mesh.Y[i, -1] + mesh.Y[i-1, -1]) / 2

    for i in range(1, mesh.M):
        for j in range(1, mesh.N):
            base_1 = ((mesh.X[i, j-1] - mesh.X[i-1, j-1]) ** 2\
                     + (mesh.Y[i, j-1] - mesh.Y[i-1, j-1]) ** 2) ** 0.5
            base_2 = ((mesh.X[i, j] - mesh.X[i-1, j]) ** 2\
                     + (mesh.Y[i, j] - mesh.Y[i-1, j]) ** 2) ** 0.5
            height = ()


    plt.figure('aspect')
    plt.axis('equal')
    plt.plot(mesh.X, mesh.Y, 'k')
    plt.plot(mesh.X[:, 0], mesh.Y[:, 0], 'k')
    for i in range(mesh.M):
        plt.plot(mesh.X[i, :], mesh.Y[i, :], 'b')

    plt.plot(X, Y, '*g')
    plt.plot(X[:, 0], Y[:, 0], '*c')
    for i in range(mesh.M):
        plt.plot(X[i, :], Y[i, :], '*r')

    plt.draw()
    plt.show()

