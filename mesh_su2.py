#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:53:21 2018

@author: Cardoso

Scripts para convertir mallas a formato de SU2
"""

import numpy as np

def to_su2_mesh_o_airfoil(mesh):
    '''
    Convierte malla de formato propio a formato de SU2
    Para mallas tipo O
    Con sólo un perfil (o cualquier geometría)
    '''

    # importa coordenadas de mesh, se quita ultima columna.
    X = np.copy(mesh.X)[:-1, :]
    Y = np.copy(mesh.Y)[:-1, :]

    # convirtiendo a arreglo 1D
    X = X.flatten()
    Y = Y.flatten()

    print(np.shape(X))
    print(np.shape(mesh.X.flatten()))
    exit()
