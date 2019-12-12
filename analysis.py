#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 26 2019

@author: cardoso

"""

import numpy as np
import matplotlib.pyplot as plt
import time


def potential_flow_o(d0, H0, gamma, mach_inf, v_inf, alfa, mesh):
    '''
        cálculo de flujo potencial para una malla tipo O
        d = densidad
        Las variables que terminan en H o V corresponden a los valores de
            dichas variables en las mallas intercaladas.
            H: malla horizontal (intercalada en [i +- 1/2, j])
            V: malla vertical (intercalada en [i, j +- 1/2])
    '''
    # se definen variables de la malla
    X = np.copy(mesh.X)
    Y = np.copy(mesh.Y)
    M = mesh.M
    N = mesh.N
    d_xi = mesh.d_xi
    d_eta = mesh.d_eta

    # se definen matrices de valores en mallas intercaladas
    '''
    x_xiV = np.zeros((M, N-1))
    x_etaV = np.zeros((M, N-1))
    y_xiV = np.zeros((M, N-1))
    y_etaV = np.zeros((M, N-1))
    g11V = np.zeros((M, N-1))
    g12V = np.zeros((M, N-1))
    g22V = np.zeros((M, N-1))
    JV = np.zeros((M, N-1))

    x_xiH = np.zeros((M-1, N))
    x_etaH = np.zeros((M-1, N))
    y_xiH = np.zeros((M-1, N))
    y_etaH = np.zeros((M-1, N))
    g11H = np.zeros((M-1, N))
    g12H = np.zeros((M-1, N))
    g22H = np.zeros((M-1, N))
    JH = np.zeros((M-1, N))
    '''
    x_xiV = np.zeros((M, N))
    x_etaV = np.zeros((M, N))
    y_xiV = np.zeros((M, N))
    y_etaV = np.zeros((M, N))
    g11V = np.zeros((M, N))
    g12V = np.zeros((M, N))
    g22V = np.zeros((M, N))
    JV = np.zeros((M, N))

    x_xiH = np.zeros((M, N))
    x_etaH = np.zeros((M, N))
    y_xiH = np.zeros((M, N))
    y_etaH = np.zeros((M, N))
    g11H = np.zeros((M, N))
    g12H = np.zeros((M, N))
    g22H = np.zeros((M, N))
    JH = np.zeros((M, N))
    # calculo del tensor de la métrica
    (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, _, _, _) = mesh.tensor()

    # cálculo de términos en mallas intercaladas promediando los valores de los
    # nodos vecinos
    # malla vertical
    for j in range(N-1):
        g11V[:, j] = 0.5 * (g11[:, j] + g11[:, j+1])
        g12V[:, j] = 0.5 * (g12[:, j] + g12[:, j+1])
        g22V[:, j] = 0.5 * (g22[:, j] + g22[:, j+1])
        JV[:, j] = 0.5 * (J[:, j] + J[:, j+1])
        x_xiV[:, j] = 0.5 * (x_xi[:, j] + x_xi[:, j+1])
        x_etaV[:, j] = 0.5 * (x_eta[:, j] + x_eta[:, j+1])
        y_xiV[:, j] = 0.5 * (y_xi[:, j] + y_xi[:, j+1])
        y_etaV[:, j] = 0.5 * (y_eta[:, j] + y_eta[:, j+1])

    ###########################################################################
    #
    #   g11V coinicide en todo, excepto en j = 0 y i = 0 y i = -1
    #   g12V coinicide en todo, excepto en j = 0 y i = 0 y i = -1
    #   g22V coinicide en todo CASI PERFECTAMENTE. en j = 0 hay ligeras
    #       variaciones
    #   JV coincide casi PERFECTAMENTE
    #   x_xiV coincide casi PERFECTAMENTE
    #   x_etaV coincide casi PERFECTAMENTE
    #   y_xiV coincide casi PERFECTAMENTE
    #   y_etaV coincide casi PERFECTAMENTE
    #
    ###########################################################################

    ###########################################################################
    #
    #   La forma de las matrices g11V, g12V, g22V, JV en MI código es (35, 34)
    #       mientras que en el código de la españoleta es de (34, 34)
    #   La forma de las matrices x_xiV, y_xiV, x_etaV, y_etaV en MI código es
    #       (35, 34) mientras que en el código de la españoleta es de (35, 35)
    #
    #   No parece tener mucho sentido lo que hizo la españoleta ya que en su
    #       PDF dice que la malla vertical es de tamaño (M, N-1)
    #
    ###########################################################################

    # malla horizontal
    for i in range(M-1):
        g11H[i, :] = 0.5 * (g11[i, :] + g11[i+1, :])
        g12H[i, :] = 0.5 * (g12[i, :] + g12[i+1, :])
        g22H[i, :] = 0.5 * (g22[i, :] + g22[i+1, :])
        JH[i, :] = 0.5 * (J[i, :] + J[i+1, :])
        x_xiH[i, :] = 0.5 * (x_xi[i, :] + x_xi[i+1, :])
        x_etaH[i, :] = 0.5 * (x_eta[i, :] + x_eta[i+1, :])
        y_xiH[i, :] = 0.5 * (y_xi[i, :] + y_xi[i+1, :])
        y_etaH[i, :] = 0.5 * (y_eta[i, :] + y_eta[i+1, :])
    ###########################################################################
    #
    #   g11H coincide casi PERFECTAMENTE. Algunas inconsistencias en i = 0
    #       e i = -1, tanto para las primeras como últimas j. NADA GRAVE
    #
    #   g12H coincide casi PERFECTAMENTE. algunas diferencias, nada grave
    #   g22H coincide casi PERFECTAMENTE.
    #   x_xiH coincide casi PERFECTAMENTE.
    #   x_etaH coincide casi PERFECTAMENTE.
    #   y_xiH coincide casi PERFECTAMENTE.
    #   y_etaH coincide casi PERFECTAMENTE.
    ###########################################################################

    ###########################################################################
    #
    #   La forma de las matrices g11H, g12H, g22H, JH en MI código es (34, 35)
    #       mientras que en el código de la españoleta es de (34, 34)
    #   La forma de las matrices x_xiH, y_xiH, x_etaH, y_etaH en MI código es
    #       (34, 35) mientras que en el código de la españoleta es de (35, 35)
    #
    #   No parece tener mucho sentido lo que hizo la españoleta ya que en su
    #       PDF dice que la malla vertical es de tamaño (M-1, N)
    #
    ###########################################################################

    # se calcula el ángulo theta de cada nodo, resultado en ángulos absolutos
    # desde 0 hasta 2 * pi
    theta = np.arctan2(Y, X)
    mask = theta < 0
    theta[mask] += 2 * np.pi
    theta[0, :] = 2 * np.pi
    theta[-1, :] = 0
    ###########################################################################
    #
    #   theta coincide PERFECTAMENTE
    #
    ###########################################################################

    # convierte angulo de ataque a radianes
    alfa = alfa * np.pi / 180
    ###########################################################################
    #
    #   Inicia proceso iterativo para la solución del flujo potencial
    #
    ###########################################################################
    C = 0.5
    phi = np.zeros((M, N))
    '''
    UH = np.zeros((M-1, N))
    VH = np.zeros((M-1, N))
    UV = np.zeros((M, N-1))
    VV = np.zeros((M, N-1))
    PH = np.zeros((M-1, N))
    PV = np.zeros((M, N-1))
    '''
    UH = np.zeros((M, N))
    VH = np.zeros((M, N))
    UV = np.zeros((M, N))
    VV = np.zeros((M, N))
    PH = np.zeros((M, N))
    PV = np.zeros((M, N))
    ddd = 100
    it_max = 100  # 2000  # 20000
    it = 0
    error = 1e-9
    omega = 0.5
    IMA = 0

    arcotan = np.zeros((M,))
    arcosen = np.zeros((M,))

    arcotan[:] = np.arctan((1 - mach_inf ** 2) ** 0.5
                           * np.tan(theta[:, 0] - alfa))
    ###########################################################################
    #
    #   arcotan coincide CASI PERFECTAMENTE. Hay diferencias cerca de las
    #       fronteras i = 0 e i = -1. conforme uno se acerca al 'centro' van
    #       coincidiendo mejor los valores
    #   arcosen coincide CASI PERFECTAMENTE. Hay diferencias cerca de las
    #       fronteras i = 0 e i = -1. conforme uno se acerca al 'centro' van
    #       coincidiendo mejor los valores
    #
    ###########################################################################
    arcosen[:] = np.arcsin((1 - mach_inf ** 2) ** 0.5
                           * np.sin(theta[:, 0] - alfa))
    for i in range(M):
        if arcotan[i] > 0 and arcosen[i] < 0:
            arcotan[i] += np.pi
        elif arcotan[i] < 0 and arcosen[i] > 0:
            arcotan[i] = np.pi - np.abs(arcotan[i])
        elif arcotan[i] < 0 and arcosen[i] < 0:
            if theta[i, -1] - alfa > 0:
                arcotan[i] += (2 * np.pi)
    ###########################################################################
    #
    #   arcotan coincide casi PERFECTAMENTE
    #
    ###########################################################################

    print('Potential Flow')
    while ddd > error and it < it_max:
        print(it, end='\r')
        it += 1
        phi_old = np.copy(phi)

        # Función potencial en la frontera externa
        #  phi[:, -1] = v_inf * (X[:, -1] * np.cos(alfa) + Y[:, -1]
        #                        * np.sin(alfa)) + C * arcotan[:] / 2 / np.pi
        phi[:, 0] = v_inf * (X[:, 0] * np.cos(alfa) + Y[:, 0]
                              * np.sin(alfa)) + C * arcotan[:] / 2 / np.pi
        '''
        it == 1
        phi coincide perfectamente
        '''

        # Nodos internos de la malla
        # velocidades U y V en mallas intercaladas V y H (vertical, horizontal)
        # malla vertical
        #  for j in range(N-1):
        #      PV[0, j] = 0.25 * (phi[1, j+1] - phi[-2, j+1]
        #                          + phi[1, j] - phi[-2, j] + 2 * C)  # 2c?
        #      for i in range(1, M-1):
        #          PV[i, j] = 0.25 * (phi[i+1, j+1] - phi[i-1, j+1]
        #                             + phi[i+1, j] - phi[i-1, j])
        for i in range(M-1):
            for j in range(N-1):
                if i == 0 and j == N-2:
                    PV[i, j] = 0.25 * (phi[i+1, j] - phi[M-2, j]
                            + phi[i+1, j-1] - phi[M-2, j-1] + 2 * C)
                elif i == 0 and j != N-2:
                    PV[i, j] = 0.25 * (phi[i+1, j+2] - phi[M-2, j+2]
                            + phi[i+1, j+1] - phi[M-2, j+1] + 2 * C)
                elif i != 0 and j == N-2:
                    PV[i, j] = 0.25 * (phi[i+1, j] - phi[i-1, j]
                            + phi[i+1, j-1] - phi[i-1, j-1])
                else:
                    PV[i, j] = 0.25 * (phi[i+1, j+2] - phi[i-1, j+2]
                            + phi[i+1, j+1] - phi[i-1, j+1])
                if j == N-2:
                    UV[i, j] = g11V[i, j] * PV[i, j] + g12V[i, j] * (phi[i, j]
                            - phi[i, j-1])
                    VV[i, j] = g12V[i, j] * PV[i, j] + g22V[i, j] * (phi[i, j]
                            - phi[i, j-1])
                else:
                    UV[i, j] = g11V[i, j] * PV[i, j] + g12V[i, j] \
                            * (phi[i, j+2] - phi[i, j +1])
                    VV[i, j] = g12V[i, j] * PV[i, j] + g22V[i, j] \
                            * (phi[i, j+2] - phi[i, j +1])
        '''
        it == 1
            PV coincide PERFECTAMENTE
            UV coincide casi PERFECTAMENTE excepto en [0, 0]
            VV coincide casi PERFECTAMENTE excepto en [0, :]
        '''
        #  for j in range(N-1):
        #      UV[:, j] = g11V[:, j] * PV[:, j] + g12V[:, j] \
        #              * (phi[:, j+1] - phi[:, j])
        #      VV[:, j] = g12V[:, j] * PV[:, j] + g22V[:, j] \
        #          * (phi[:, j+1] - phi[:, j])
        #  for j in range(N-2):
        #      UVp[:M-2, j] = g11V[i, j] * PV[i, j] + g12V[i, j] * (phi[i, j+2]
        #              - phi[i, j+1])
        #      VVp[:M-2, j] = g12V[i, j] * PV[i, j] + g22V[i, j] * (phi[i, j+2]
        #              - phi[i, j+1])
        #  UVp[:M-2, N-2] = g11V[i, j] * PV[i, j] + g12V[i, j] * (phi[i, j]
        #          - phi[i, j-1])
        #  VVp[:M-2, N-2] = g12V[i, j] * PV[i, j] + g22V[i, j] * (phi[i, j]
        #          - phi[i, j-1])
        # malla horizontal
        #  for i in range(M-1):
        #      PH[i, -1] = 0.5 * (phi[i+1, -1] - phi[i+1, -2] + phi[i, -1]
        #                         - phi[i, -2])
        #      PH[i, 0] = 0.5 * (phi[i+1, 1] - phi[i+1, 0] + phi[i, 1]
        #                        - phi[i, 0])
        #      for j in range(1, N-2):
        #          PH[i, j] = 0.25 * (phi[i+1, j+1] - phi[i+1, j-1]
        #                             + phi[i, j+1] - phi[i, j-1])
        for i in range(M-1):
            for j in range(1, N-1):
                PH[i, j] = 0.25 * (phi[i+1, j+1] - phi[i+1, j-1] + phi[i, j+1]
                        - phi[i, j-1])
        for i in range(M-1):
            UH[i, 1 : N-1] = g11H[i, 1 : N-1] * (phi[i+1, 1 : N-1] - phi[i, 1:N-1]) + g12H[i, 1:N-1] \
                        * PH[i, 1:N-1]
            VH[i, 1 : N-1] = g12H[i, 1:N-1] * (phi[i+1, 1:N-1] - phi[i, 1:N-1]) + g22H[i, 1:N-1] \
                * PH[i, 1:N-1]
        '''
        it = 1
        PH coincide PERFECTAMENTE
        UH coincide PERFECTAMENTE
        VH coincide PERFECTAMENTE
        '''

        rhoV = 1 - (UV**2 * (x_xiV**2 + y_xiV**2)
                    + VV**2 * (x_etaV**2 + y_etaV**2)
                    + 2 * UV * VV * (x_xiV * x_etaV + y_xiV * y_etaV))\
                    / 2 / H0
        rhoH = 1 - ((UH**2 * (x_xiH**2 + y_xiH**2)
                    + VH**2 * (x_etaH**2 + y_etaH**2)
                    + 2 * UH * VH * (x_xiH * x_etaH + y_xiH * y_etaH))
                    / 2 / H0)

        # checando los valores de la densidad
        IMA = 0
        mV = rhoV < 0
        mH = rhoH < 0
        if mV.any() or mH.any():
            IMA = 1
        rhoV_tmp = np.copy(rhoV)
        rhoH_tmp = np.copy(rhoH)

        rhoV = d0 * np.abs(rhoV_tmp) ** (1 / (gamma - 1))  # abs()?
        rhoH = d0 * np.abs(rhoH_tmp) ** (1 / (gamma - 1))  # abs()?
        '''
        it = 1
        rhoV coincide en todo except en [0, 0]. VARIA MUCHO EL VALOR
        rhoH coincide PERFECTAMENTE
        '''

        #######################################################################
        #
        #   YA SE HA REVISADO EL CÁLCULO DE PHI Y PARECE ESTAR BIEN
        #       hubo mejora, de it = 5 a it = 14
        #
        #######################################################################
        # cálculo de función potencial phi
        '''
        for i in range(M-1):
            for j in range(1, N-1):
            # para i = 0
                if i == 0:
                    phi[0, j] = 1 / (rhoH[0, j] * JH[0, j] * g11H[0, j]
                              + rhoH[M-2, j] * JH[M-2, j] * g11H[M-2, j]
                              + rhoV[0, j] * JV[0, j] * g22V[0, j]
                              + rhoV[0, j-1] * JV[0, j-1] * g22V[0, j-1]) \
                * \
                (rhoH[0, j] * JH[0, j] * (g12H[0, j] * PH[0, j]
                                            + g11H[0, j] * phi[1, j])
                    - rhoH[M-2, j] * JH[M-2, j] * (g12H[M-2, j] * PH[M-2, j]
                                                 - g11H[M-2, j] * (phi[M-2, j] - C))
                    + rhoV[0, j-1] * JV[0, j-1] * (g12V[0, j-1] * PV[0, j-1]
                                                 + g22V[0, j-1] * phi[0, j])
                    - rhoV[0, j] * JV[0, j] * (g12V[0, j]
                                * PV[0, j] - g22V[0, j] * phi[0, j-1]))
            # for i in range(1, M-1):
                else:
                    phi[i, j] = 1 / (rhoH[i, j] * JH[i, j] * g11H[i, j]
                                 + rhoH[i-1, j] * JH[i-1, j] * g11H[i-1, j]
                                 + rhoV[i, j] * JV[i, j] * g22V[i, j]
                                 + rhoV[i, j-1] * JV[i, j-1] * g22V[i, j-1]) \
                                 * (rhoH[i, j] * JH[i, j] * (g12H[i, j]
                                    * PH[i, j] + g11H[i, j] * phi[i+1, j])
                                    - rhoH[i-1, j] * JH[i-1, j]
                                    * (g12H[i-1, j] * PH[i-1, j] - g11H[i-1, j]
                                        * phi[i-1, j])
                                    - rhoV[i, j] * JV[i, j] * (g12V[i, j]
                                        * PV[i, j] - g22V[i, j] * phi[i, j-1])
                                    + rhoV[i, j-1] * JV[i, j-1] * (g12V[i, j-1]
                                        * PV[i, j-1] + g22V[i, j-1]
                                        * phi[i, j+1]))
                phi[i, j] = omega * phi[i, j] + (1 - omega) * phi[i, j]
        '''
        # for i=1:N-1:
        for i in range(M-1):
            # for j=2:M-1:
            for j in range(1, N-1):
                if i == 0:
                    phi[i,j] = (rhoH[i,j] * JH[i,j] * (g12H[i,j] * PH[i,j] \
                            + g11H[i,j] * phi[i+1,j]) - rhoH[N-1,j] * JH[M-2,j]\
                            * (g12H[M-2,j] * PH[M-2,j] - g11H[M-2,j] \
                            * (phi[M-2,j] - C)) + rhoV[i,j-1] * JV[i,j-1] \
                            * (g12V[i,j-1] * PV[i,j-1] + g22V[i,j-1] \
                            * phi[i,j]) - rhoV[i,j] * JV[i,j] * (g12V[i,j] \
                            * PV[i,j] - g22V[i,j] * phi[i,j-1])) \
                            / (rhoH[i,j] * JH[i,j] * g11H[i,j] + rhoH[M-2,j] \
                            * JH[M-2,j] * g11H[M-2,j] + rhoV[i,j] * JV[i,j] \
                            * g22V[i,j] + rhoV[i,j-1] * JV[i,j-1] * g22V[i,j-1])
                else:
                    phi[i,j] = (rhoH[i,j] * JH[i,j] * (g12H[i,j] * PH[i,j] \
                            + g11H[i,j] * phi[i+1,j]) - rhoH[i-1,j] * JH[i-1,j]\
                            * (g12H[i-1,j] * PH[i-1,j] - g11H[i-1,j]\
                            * (phi[i-1,j])) \
                            + rhoV[i,j-1] * JV[i,j-1] \
                            * (g12V[i,j-1] * PV[i,j-1] + g22V[i,j-1] \
                            * phi[i,j+1]) - rhoV[i,j] * JV[i,j] * (g12V[i,j] \
                            * PV[i,j] - g22V[i,j] * phi[i,j-1])) \
                            / (rhoH[i,j] * JH[i,j] * g11H[i,j] + rhoH[i-1,j] \
                            * JH[i-1,j] * g11H[i-1,j] + rhoV[i,j] * JV[i,j] \
                            * g22V[i,j] + rhoV[i,j-1] * JV[i,j-1] * g22V[i,j-1])

        # Aplicamos el m�todo SOR de sobrerelajaci�n, ecuaci�n (4.29).
        # phi(i,j) = w * phi(i,j) + (1-w) * phiold(i,j);
        phi = omega * phi + (1-omega) * phi_old
        print('phi[8, :]')
        print(phi[8, :])
        exit()

        # condición en la superficie del perfil

        #######################################################################
        #
        #   YA SE HA HECHO LA CONVERSION ENTRE ESPAÑOLETA Y YO
        #
        #######################################################################
        # for i in range(1, M-1):
        for i in range(M-2, 0, -1):
            phi[i, N-1] = 1 / 3 * (4 * phi[i, N-2] - phi[i, N-3] - g12[i, N-2]
                            / g22[i, N-2] * (phi[i+1, N-1] - phi[i-1, N-1]))


        phi[0, N-1] = 1 / 3 * (4 * phi[0, N-2] - phi[0, N-3] - g12[0, N-1]
                            / g22[0, N-1] * (phi[1, N-1] - phi[M-2, N-1] + C))

        # discontinuidad del potencial
        # print(phi[0, :])
        # exit()

        ddd = abs(phi - phi_old).max()

        # cálculo de la Circulación
        #######################################################################
        #
        #   YA SE HA HECHO LA CONVERSION ENTRE ESPAÑOLETA Y YO
        #
        #######################################################################
        C = phi[M-2, N-1] - phi[1, N-1] - g12[0, N-1] * \
            (phi[0, N-3] - 4 * phi[0, N-2] + 3 * phi[0, N-1]) / g11[0, N-1]
    print('outside while. it = ' + str(it))


'''
    CÓDIGO ESPAÑOLETA
'''


def potential_flow_o_esp(d0, H0, gamma, mach_inf, v_inf, alfa, mesh):
    '''
        cálculo de flujo potencial para una malla tipo M
        d = densidad
        Las variables que terminan en H o V corresponden a los valores de
            dichas variables en las mallas intercaladas.
            H: malla horizontal (intercalada en [i +- 1/2, j])
            V: malla vertical (intercalada en [i, j +- 1/2])
    '''

    # se definen variables de la malla
    X = np.copy(mesh.X)
    Y = np.copy(mesh.Y)
    M = mesh.M
    N = mesh.N
    # d_xi = mesh.d_xi
    # d_eta = mesh.d_eta

    (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, _, _, _) = \
        mesh.tensor_esp()
    print('after tensor')

    # importing from ESPAÑOLETA para LINUX
    # g11e = np.genfromtxt('/home/cardoso/garbage/g11.csv', delimiter=',')
    # g12e = np.genfromtxt('/home/cardoso/garbage/g12.csv', delimiter=',')
    # g22e = np.genfromtxt('/home/cardoso/garbage/g22.csv', delimiter=',')
    # Je = np.genfromtxt('/home/cardoso/garbage/J.csv', delimiter=',')
    # x_xie = np.genfromtxt('/home/cardoso/garbage/x_xi.csv', delimiter=',')
    # x_etae = np.genfromtxt('/home/cardoso/garbage/x_eta.csv', delimiter=',')
    # y_xie = np.genfromtxt('/home/cardoso/garbage/y_xi.csv', delimiter=',')
    # y_etae = np.genfromtxt('/home/cardoso/garbage/y_eta.csv', delimiter=',')
    # Xe = np.genfromtxt('/home/cardoso/garbage/X.csv', delimiter=',')
    # Ye = np.genfromtxt('/home/cardoso/garbage/Y.csv', delimiter=',')

    # importing from ESPAÑOLETA para LINUX work
    g11e = np.genfromtxt('/home/vivoxie/garbage/g11.csv', delimiter=',')
    g12e = np.genfromtxt('/home/vivoxie/garbage/g12.csv', delimiter=',')
    g22e = np.genfromtxt('/home/vivoxie/garbage/g22.csv', delimiter=',')
    Je = np.genfromtxt('/home/vivoxie/garbage/J.csv', delimiter=',')
    x_xie = np.genfromtxt('/home/vivoxie/garbage/x_xi.csv', delimiter=',')
    y_etae = np.genfromtxt('/home/vivoxie/garbage/x_eta.csv', delimiter=',')
    y_xie = np.genfromtxt('/home/vivoxie/garbage/y_xi.csv', delimiter=',')
    y_etae = np.genfromtxt('/home/vivoxie/garbage/y_eta.csv', delimiter=',')
    Xe = np.genfromtxt('/home/vivoxie/garbage/X.csv', delimiter=',')
    Ye = np.genfromtxt('/home/vivoxie/garbage/Y.csv', delimiter=',')

    # importing from ESPAÑOLETA para MacOS
    # g11e = np.genfromtxt('/Users/cardosom/garbage/g11.csv', delimiter=',')
    # g12e = np.genfromtxt('/Users/cardosom/garbage/g12.csv', delimiter=',')
    # g22e = np.genfromtxt('/Users/cardosom/garbage/g22.csv', delimiter=',')
    # Je = np.genfromtxt('/Users/cardosom/garbage/J.csv', delimiter=',')
    # x_xie = np.genfromtxt('/Users/cardosom/garbage/x_xi.csv', delimiter=',')
    # x_etae = np.genfromtxt('/Users/cardosom/garbage/x_eta.csv', delimiter=',')
    # y_xie = np.genfromtxt('/Users/cardosom/garbage/y_xi.csv', delimiter=',')
    # y_etae = np.genfromtxt('/Users/cardosom/garbage/y_eta.csv', delimiter=',')
    # Xe = np.genfromtxt('/Users/cardosom/garbage/X.csv', delimiter=',')
    # Ye = np.genfromtxt('/Users/cardosom/garbage/Y.csv', delimiter=',')
    # g21 = g12

    index = -3
    percent = 10
    var = Y
    vare = Ye
    var += 1e-45
    vare += 1e-45
    print('inside potential')
    print('inside potential')
    print(np.all(np.abs(var - vare) / var * 100 <= percent))
    print(np.all(np.abs(vare - var) / vare * 100 <= percent))
    print('inside POTENTIAL END')
    exit()
    '''
    g12 el error es mayor a 1% pero menor a 2%
    ERRORES CATASTRÓFICOS EN Y_XI Y Y_ETA
    '''

    x_xiV = np.zeros((M, N))
    x_etaV = np.zeros((M, N))
    y_xiV = np.zeros((M, N))
    y_etaV = np.zeros((M, N))
    x_xiH = np.zeros((M, N))
    x_etaH = np.zeros((M, N))
    y_xiH = np.zeros((M, N))
    y_etaH = np.zeros((M, N))

    g11V = np.zeros((M-1, N-1))
    g12V = np.zeros((M-1, N-1))
    g22V = np.zeros((M-1, N-1))
    JV = np.zeros((M-1, N-1))

    g11H = np.zeros((M-1, N))
    g12H = np.zeros((M-1, N))
    g22H = np.zeros((M-1, N))
    JH = np.zeros((M-1, N))

    # for i=1:M-1
    for i in range(M-1):
        # for j=1:N-1
        for j in range(N-1):
            g11V[i, j] = 0.5 * (g11[i, j] + g11[i, j+1])
            g12V[i, j] = 0.5 * (g12[i, j] + g12[i, j+1])
            g22V[i, j] = 0.5 * (g22[i, j] + g22[i, j+1])
            JV[i, j] = 0.5 * (J[i, j] + J[i, j+1])
            x_xiV[i, j] = 0.5 * (x_xi[i, j] + x_xi[i, j+1])
            x_etaV[i, j] = 0.5 * (x_eta[i, j] + x_eta[i, j+1])
            y_xiV[i, j] = 0.5 * (y_xi[i, j] + y_xi[i, j+1])
            y_etaV[i, j] = 0.5 * (y_eta[i, j] + y_eta[i, j+1])

    # for i=1:M-1
    for i in range(M-1):
        # for j=1:N
        for j in range(N):
            g11H[i, j] = 0.5 * (g11[i, j] + g11[i+1, j])
            g12H[i, j] = 0.5 * (g12[i, j] + g12[i+1, j])
            g22H[i, j] = 0.5 * (g22[i, j] + g22[i+1, j])
            JH[i, j] = 0.5 * (J[i, j] + J[i+1, j])
            x_xiH[i, j] = 0.5 * (x_xi[i, j] + x_xi[i+1, j])
            x_etaH[i, j] = 0.5 * (x_eta[i, j] + x_eta[i+1, j])
            y_xiH[i, j] = 0.5 * (y_xi[i, j] + y_xi[i+1, j])
            y_etaH[i, j] = 0.5 * (y_eta[i, j] + y_eta[i+1, j])

    g21V = g12V
    g21H = g12H
    # Calculamos el ángulo theta de cada nodo. Como el arco tangente
    # nos devuelve el valor del ángulo en los cuadrantes I y IV lo
    # recalculamos según el siguiente código.
    theta = np.arctan(Y / X)
    alfa = alfa*np.pi/180
    # for i=1:M
    for i in range(M):
        # for j=1:N
        for j in range(N):
            if X[i, j] <= 0 and Y[i, j] >= 0:
                theta[i, j] = np.pi - np.abs(theta[i, j])
            elif X[i, j] <= 0 and Y[i, j] < 0:
                theta[i, j] = np.abs(theta[i, j]) + np.pi
            elif X[i, j] > 0 and Y[i, j] < 0:
                theta[i, j] = 2 * np.pi - np.abs(theta[i, j])

    theta[M-1, :] = 2 * np.pi
    theta[0, :] = 0
    #----------------------------VALMR IMICIAL----------------------------#
    C = 0.5
    phi = np.zeros((M, N))
    UH = np.zeros((M, N))
    VH = np.zeros((M, N))
    UV = np.zeros((M, N))
    VV = np.zeros((M, N))

    PV = np.zeros((M-1, N-1))
    PH = np.zeros((M-1, N-1))

    DDV = np.zeros((M, N))
    DDH = np.zeros((M, N))
    dV = np.zeros((M, N))
    dH = np.zeros((M, N))

    arcotan = np.zeros((M,))
    arcosen = np.zeros((M,))
    it = 0
    ddd = 1
    it_max = 20000
    tol = 1.e-9
    omega = 0.1
    while ddd > tol and it < it_max:
        it = it + 1
        print(it, end='\r')
        phi_old = np.copy(phi)

        # -------------------------FRMMTERA EXTERIMR--------------------------#
        # Para aplicar la fórmula (2.30) primero determinamos el arco tangente
        # y lo distribuimos igual que en el caso de theta.
        arcotan[:] = np.arctan(np.sqrt(1 - mach_inf ** 2) \
                        * np.tan(theta[:, 1] - alfa))
        arcosen[:] = np.arcsin(np.sqrt(1 - mach_inf ** 2) \
                        * np.sin(theta[:, 1] - alfa))

        # for i=1:M
        for i in range(M):
            if arcotan[i] > 0 and arcosen[i] < 0:
                arcotan[i] = arcotan[i] + np.pi
            elif arcotan[i] < 0 and arcosen[i] > 0:
                arcotan[i] = np.pi - np.abs(arcotan[i])
            elif arcotan[i] < 0 and arcosen[i] < 0:
                if (theta[i, 0] - alfa) > 0:
                    arcotan[i] = 2 * np.pi + arcotan[i]

        phi[:, 0] = v_inf * (X[:, 0] * np.cos(alfa) + Y[:, 0] \
                        * np.sin(alfa)) + C * arcotan[:] / (2 * np.pi)

        # --------------------MMDMS IMTERMMS DE LA NALLA----------------------#
        # Desarrollamos los parámetros en los nodos intercalados desarrollando
        # las fórmulas (4.9) y (4.10).
        # for i=1:M-1
        for i in range(M-1):
            # for j=1:N-1
            for j in range(N-1):
                if i == 0 and j == N-2:
                    PV[i, j] = 0.25 * (phi[i+1, j] - phi[M-2, j] \
                            + phi[i+1, j-1] - phi[M-2, j-1] + 2 * C)
                elif i == 0 and j != N-2:
                    PV[i, j] = 0.25 * (phi[i+1, j+2] - phi[M-2, j+2] \
                            + phi[i+1, j+1] - phi[M-2, j+1] + 2 * C)
                elif i != 0 and j == N-2:
                    PV[i, j] = 0.25 * (phi[i+1, j] - phi[i-1, j] \
                            + phi[i+1, j-1] - phi[i-1, j-1])
                else:
                    PV[i, j] = 0.25 * (phi[i+1, j+2] - phi[i-1, j+2] \
                            + phi[i+1, j+1] - phi[i-1, j+1])

                if j == N-2:
                    UV[i, j] = g11V[i, j] * PV[i, j] + g12V[i, j] \
                            * (phi[i, j] - phi[i, j-1])
                    VV[i, j] = g21V[i, j] * PV[i,j] + g22V[i,j] \
                            * (phi[i, j] - phi[i, j-1])
                else:
                    UV[i, j] = g11V[i, j] * PV[i, j] + g12V[i, j] \
                            * (phi[i, j+2] - phi[i, j+1])
                    VV[i, j] = g21V[i, j] * PV[i, j] + g22V[i, j] \
                            * (phi[i, j+2] - phi[i, j+1])

        # for i=1:M-1
        for i in range(M-1):
            # for j=2:N-1
            for j in range(1, N-1):
                PH[i, j] = 0.25 * (phi[i+1, j+1] - phi[i+1, j-1] \
                        + phi[i, j+1] - phi[i, j-1])
                UH[i, j] = g11H[i, j] * (phi[i+1, j] - phi[i, j]) \
                        + g12H[i, j] * PH[i, j]
                VH[i, j] = g21H[i, j] * (phi[i+1, j] - phi[i, j]) \
                        + g22H[i, j] * PH[i, j]

        # Calculamos la densidad, ecuación (4.13)
        IMA = 0
        # for i=1:M
        for i in range(M):
            # for j=1:N
            for j in range(N):
                DDV[i, j] = 1 - ((x_xiV[i, j] ** 2 + y_xiV[i, j] ** 2) \
                        * UV[i, j] ** 2 + (x_etaV[i, j] ** 2 \
                        + y_etaV[i, j] ** 2) * VV[i, j] ** 2 + 2 * UV[i,j] \
                        * VV[i, j] * (x_xiV[i, j] * x_etaV[i, j] \
                        + y_xiV[i, j] * y_etaV[i, j])) / (2 * H0)
                DDH[i, j] = 1 - ((x_xiH[i, j] ** 2 + y_xiH[i, j] ** 2) \
                        * UH[i, j] ** 2 + (x_etaH[i, j] ** 2 \
                        + y_etaH[i, j] ** 2) * VH[i, j] ** 2 + 2 * UH[i, j] \
                        * VH[i, j] * (x_xiH[i, j] * x_etaH[i, j] \
                        + y_xiH[i, j] * y_etaH[i, j])) / (2 * H0)
                if DDV[i, j] < 0 or DDH[i, j] < 0:
                    IMA = 1

                dV[i, j] = d0 * np.abs(DDV[i, j]) ** (1 / (gamma - 1))
                dH[i, j] = d0 * np.abs(DDH[i, j]) ** (1 / (gamma - 1))

        # Introducimos las variables anteriores en la ecuación del potencial.
        # Fórmula (4.11)
        # for i=1:M-1
        for i in range(M-1):
            # for j=2:N-1
            for j in range(1, N-1):
                if i == 0:
                    phi[i, j] = (dH[i, j] * JH[i, j] * (g12H[i, j] * PH[i, j] \
                            + g11H[i, j] * phi[i+1, j]) - dH[M-2, j] \
                            * JH[M-2, j] * (g12H[M-2, j] * PH[M-2, j] \
                            - g11H[M-2, j] * (phi[M-2, j] - C)) + dV[i, j-1] \
                            * JV[i, j-1] * (g21V[i, j-1] * PV[i, j-1] \
                            + g22V[i, j-1] * phi[i, j]) - dV[i, j] * JV[i, j] \
                            * (g21V[i, j] * PV[i, j] - g22V[i, j] \
                            * phi[i, j-1])) / (dH[i, j] * JH[i, j] \
                            * g11H[i, j] + dH[M-2, j] * JH[M-2, j] \
                            * g11H[M-2, j] + dV[i, j] * JV[i, j] * g22V[i, j] \
                            + dV[i, j-1] * JV[i, j-1] * g22V[i, j-1])
                else:
                    phi[i, j] = (dH[i, j] * JH[i, j] * (g12H[i, j] * PH[i, j] \
                            + g11H[i, j] * phi[i+1, j]) - dH[i-1, j] \
                            * JH[i-1, j] * (g12H[i-1, j] * PH[i-1, j] \
                            - g11H[i-1, j] * (phi[i-1, j])) + dV[i, j-1] \
                            * JV[i, j-1] * (g21V[i, j-1] * PV[i, j-1] \
                            + g22V[i, j-1] * phi[i, j+1]) - dV[i,j] * JV[i,j] \
                            * (g21V[i, j] * PV[i, j] - g22V[i, j] \
                            * phi[i, j-1])) / (dH[i, j] * JH[i, j] \
                            * g11H[i, j] + dH[i-1, j] * JH[i-1, j] \
                            * g11H[i-1, j] + dV[i, j] * JV[i, j] * g22V[i, j] \
                            + dV[i, j-1] * JV[i, j-1] * g22V[i,  j-1])
                # Aplicamos el método SMR de sobrerelajación, ecuación (4.29).
                phi[i, j] = omega * phi[i, j] + (1 - omega) * phi_old[i, j]

        g21 = g12

        # --------------------CONDICIÓN EN LA SUPEFICIE-----------------------#
        # Aplicamos la fórmula (4.15)
        # for i=M-1:-1:2
        for i in range(M-2, 0, -1):
            phi[i, N-1] = (1 / 3) * (4 * phi[i, N-2] - phi[i, N-3] \
                    - g21[i, j] * (phi[i+1, N-1] - phi[i-1, N-1]) / g22[i, j])

        phi[0, N-1] = (1 / 3) * (4 * phi[1, N-2] - phi[1, N-3] - g21[0, N-1] \
                * (phi[1, N-1] - phi[M-2, N-1] + C) / g22[0, N-1])

        # ------------CMMDICIÓM EM LA DISCMMTIMUIDAD DEL PMTEMCIAL------------#
        # Aplicamos la condición de Kutta, ecuación (4.16)
        # for j=1:N
        for j in range(N):
            phi[M-1, j] = phi[0, j] + C

        ddd = np.max(np.abs(phi - phi_old))
        # ----------------------CÁLCULM DE LA CIRCULACIÓM---------------------#
        # Utilizamos la ecuación (4.18) que impone velocidad nula en el borde
        # de salida
        C = phi[M-2, N-1] - phi[1, N-1] - g12[0, N-1] \
                * (phi[0, N-3] - 4 * phi[0, N-2] + 3 * phi[0, N-1]) \
                / g11[0, N-1]
        print('it = ' + str(it))
        print('it = ' + str(it))
        print('C = ' + str(C))
        print('ddd = ' + str(ddd))
    print('outside while. it = ' + str(it))
    print('C = ' + str(C))
    print('ddd = ' + str(ddd))
    print('IMA = ' + str(IMA))
