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

    # importa coordenadas de mesh, se quita ultima columna.
    X = np.copy(mesh.X)[:-1, :].transpose()
    Y = np.copy(mesh.Y)[:-1, :].transpose()

    print(mesh.X)
    print(X)
    # convirtiendo a arreglo 1D
    X = X.flatten()
    Y = Y.flatten()
    print(X)

    NPOIN = np.shape(X)[0]

    # creando archivo de malla para SU2
    su2_mesh = open(filename, 'w')
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
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(i + 2) + '\n')

    su2_mesh.write('MARKER_TAG= airfoil\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(mesh.M - 1) + '\n')
    for i in range(mesh.M - 2):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(0) + '\n')

    return
