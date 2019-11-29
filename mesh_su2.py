#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:53:21 2018

@author: Cardoso

Scripts para convertir mallas a formato de SU2

Documentación: https://su2code.github.io/docs/Mesh-File
"""

import numpy as np
import matplotlib.pyplot as plt

def to_su2_mesh_o_airfoil(mesh, filename):
    '''
    Convierte malla de formato propio a formato de SU2
    Para mallas tipo O
    Con sólo un perfil (o cualquier geometría)
    '''

    # importa coordenadas de mesh, se quita ultima fila (repetida).
    X           = np.copy(mesh.X)[:-1, :].transpose()
    Y           = np.copy(mesh.Y)[:-1, :].transpose()

    # convirtiendo a arreglo 1D
    X           = X.flatten()
    Y           = Y.flatten()

    NPOIN       = np.shape(X)[0]

    # creando archivo de malla para SU2
    su2_mesh    = open(filename, 'w')

    su2_mesh.write('NDIME= 2\n')
    su2_mesh.write('NPOIN= ' + str(NPOIN) + '\n')

    # se escriben las coordenadas de los nodos
    for i in range(NPOIN):
        su2_mesh.write(str(X[i]) + '\t' + str(Y[i]) + '\n')

    # se escriben las celdas y la conectividad entre nodos que la forman
    NELEM = (mesh.M - 1) * (mesh.N - 1)
    su2_mesh.write('NELEM= ' + str(NELEM) + '\n')
    for i in range(NELEM):
        # condición para excluir último volumen del nivel. Al terminar vuelta
        if i % (mesh.M - 1) != mesh.M - 2:
            su2_mesh.write('9 ' + str(i) + ' ' + str(i + 1) + ' '
                       + str(i + mesh.M) + ' ' + str(i + mesh.M - 1) + '\n')
        else:
            su2_mesh.write('9 ' + str(i) + ' ' + str(i - (mesh.M - 2)) + ' '
                       + str(i + 1) + ' ' + str(i + mesh.M - 1) + '\n')

    # se escriben las fronteras. Primero FE, luego FI
    NMARK = 2
    su2_mesh.write('NMARK= ' + str(NMARK) + '\n')
    su2_mesh.write('MARKER_TAG= faraways\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(mesh.M - 1) + '\n')
    far1 = (mesh.M - 1) * mesh.N
    far0 = far1 - (mesh.M - 1)
    for i in range(far0, far1 - 1):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(far0) + '\n')

    su2_mesh.write('MARKER_TAG= airfoil\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(mesh.M - 1) + '\n')
    for i in range(mesh.M - 2):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(0) + '\n')

    return

def to_su2_mesh_o_airfoil_n_flap(mesh, filename):
    '''
    Convierte malla de formato propio a formato de SU2
    Para mallas tipo O
    Para perfiles con external airfoil flap (o en general,
        2 geometrías separadas)
    '''

    # obtiendo total de puntos de cada perfil
    size_airfoil    = 0
    size_flap       = 0
    is_boundary     = mesh.airfoil_boundary[:-1]
    print(is_boundary)
    for i in range(np.shape(is_boundary)[0]):
        if is_boundary[i] == 1:
            size_airfoil += 1
        elif is_boundary[i] == 2:
            size_flap += 1
    print('size_airfoil =', size_airfoil)
    print('size_flap=', size_flap)

    # creando archivo de malla para SU2
    su2_mesh        = open(filename, 'w')

    M_SU2           = mesh.M - 1
    N_SU2           = mesh.N - 1
    NPOIN           = M_SU2 * N_SU2 + M_SU2 - 2 - mesh.airfoil_join

    # importa coordenadas de mesh, se quita ultima fila (repetida).
    X = np.copy(mesh.X)[:-1, :]
    Y = np.copy(mesh.Y)[:-1, :]
    # extraer primera columna (perfiles) para eliminar puntos repetidos
    x_perfil        = X[:, 0]
    y_perfil        = Y[:, 0]
    end             = size_flap // 2 + 1 + mesh.airfoil_join + size_airfoil - 1
    x_perf1         = x_perfil[: end]
    y_perf1         = y_perfil[: end]
    begin           = end + mesh.airfoil_join + 2
    x_perf2         = x_perfil[begin :]
    y_perf2         = y_perfil[begin :]
    x_perfil        = np.concatenate((x_perf1, x_perf2))
    y_perfil        = np.concatenate((y_perf1, y_perf2))


    # convirtiendo a arreglo 1D
    X               = X[:, 1:]
    Y               = Y[:, 1:]
    X               = X.transpose().flatten()
    Y               = Y.transpose().flatten()
    X               = np.concatenate((x_perfil, X))
    Y               = np.concatenate((y_perfil, Y))

    # plt.plot(X, Y, '*')
    # plt.plot(X, Y)
    # plt.plot(x_perfil, y_perfil)
    # plt.plot(x_perf2, y_perf2, '*')
    # plt.axis('equal')
    # plt.draw()
    # plt.show()

    # se inicia escritura de archivo
    su2_mesh.write('NDIME= 2\n')
    su2_mesh.write('NPOIN= ' + str(NPOIN) + '\n')

    # se escriben las coordenadas de los nodos
    for i in range(NPOIN):
        su2_mesh.write(str(X[i]) + '\t' + str(Y[i]) + '\n')

    # se escriben las celdas y la conectividad entre los nodos que la forman
    NELEM = M_SU2 * N_SU2
    su2_mesh.write('NELEM= ' + str(NELEM) + '\n')

    # primera parte de celdas conectadas al perfil
    size_airfoils   = np.shape(x_perfil)[0] # numero de puntos de j[0]
    end             = size_flap // 2 + 1 + mesh.airfoil_join + size_airfoil - 2

    for i in range(end):
        su2_mesh.write('9 ' + str(i) + ' ' + str(i+1) + ' '
                       + str(i + size_airfoils + 1) + ' '
                       + str(i + size_airfoils) + '\n')

    # segunda parte, cubre el "regreso" en la O, cubre ultimo pedazo de perfil
    # y la union
    begin           = end
    extrados_flap   = end + 1
    end             += mesh.airfoil_join + 2 # 1 (union) + 1 (limite range)
    diff            = begin - size_airfoil + 2

    su2_mesh.write('9 ' + str(begin) + ' ' + str(diff) + ' '
                   + str(begin + size_airfoils + 1) + ' '
                   + str(begin + size_airfoils) + '\n')

    begin           += 1

    for i in range(begin, end):
        su2_mesh.write('9 ' + str(diff) + ' ' + str(diff - 1) + ' '
                       + str(i + size_airfoils + 1) + ' '
                       + str(i + size_airfoils) + '\n')
        diff -= 1

    # primer celda extrados flap
    begin = end
    su2_mesh.write('9 ' + str(diff) + ' ' + str(extrados_flap) + ' '
                   + str(begin + size_airfoils + 1) + ' '
                   + str(begin + size_airfoils) + '\n')
    # a partir de este punto todas las celdas siguen la misma secuencia
    # a partir de extrados de flap
    begin += 1
    diff = begin - extrados_flap
    for i in range(begin, M_SU2):
        if i % (mesh.M - 1) != mesh.M - 2:
            su2_mesh.write('9 ' + str(i - diff) + ' ' + str(i - diff + 1) + ' '
                           + str(i + size_airfoils + 1) + ' '
                           + str(i + size_airfoils) + '\n')
        else:
            if i == 35:
                print('here')
                print(i - mesh.M + 2)
            su2_mesh.write('9 ' + str(i - diff) + ' ' + str(i - mesh.M + 2)
                           + ' ' + str(i - diff + 1 ) + ' '
                           + str(i + size_airfoils) + '\n')

    begin = M_SU2
    for i in range(begin, NELEM):
        if i % (mesh.M - 1) != mesh.M - 2:
            su2_mesh.write('9 ' + str(i - diff) + ' ' + str(i - diff + 1) + ' '
                           + str(i + size_airfoils + 1) + ' '
                           + str(i + size_airfoils) + '\n')
        else:
            if i == 35:
                print('here')
                print(i - mesh.M + 2)
            su2_mesh.write('9 ' + str(i - diff) + ' '
                           + str(i - mesh.M + 2 - diff) + ' '
                           + str(i - diff + 1 ) + ' ' + str(i + size_airfoils)
                           + '\n')

    print('i = ', i)
    print('extrados_flap = ', extrados_flap )
    print('begin')
    print(begin)
    print('end')
    print(end)
    print('diff')
    print(diff)

    print('TODO')
    print('inside su2 method')
