#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    Marco Antonio Cardoso Moreno
@mail:      marcoarcardosom@gmail.com

Modulo para solucion de flujo potencial
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def potential_flow_o(d0, H0, gamma, mach_inf, v_inf, alfa, mesh):
    """
    Resuelve la ecuacion de flujo potencial.
    ...

    Parametros
    ----------
    d0 : float64
        densidad de estancamiento
    H0 : float64
        entalpia de estancamiento
    gamma : float64
        relacion de calores especificos del fluido
    mach_inf : float64
        Numero de mach de la corriente libre
    v_inf : float64
        velocidad de corriente libre
    alfa : float64
        angulo de ataque
    mesh : mesh
        Objeto mesh, que contiene toda la informacion relativa a la malla
        sobre la cual se resolvera el flujo potencial

    Return
    ------
    (phi, C, theta, IMA) : (numpy.array, float64, numpy.array, int)
        Phi: valores de la funcion de potencial en todos los nodos de la malla
        C: circulacion alrededor del perfil
        theta: angulo entre el eje X y todos los nodos de la malla
    """

    # se definen variables de la malla
    X = np.copy(mesh.X)
    Y = np.copy(mesh.Y)
    M = mesh.M
    N = mesh.N
    d_xi = mesh.d_xi
    d_eta = mesh.d_eta

    # se definen matrices de valores en mallas intercaladas
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

    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)
    X = np.copy(mesh.X)
    Y = np.copy(mesh.Y)

    # calculo del tensor de la métrica
    (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, _, _, _) = mesh.tensor()
    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)

    # cálculo de términos en mallas intercaladas promediando los valores de los
    # nodos vecinos
    # malla vertical
    # for j in range(N-1):
    #     g11V[:, j] = 0.5 * (g11[:, j] + g11[:, j+1])
    #     g12V[:, j] = 0.5 * (g12[:, j] + g12[:, j+1])
    #     g22V[:, j] = 0.5 * (g22[:, j] + g22[:, j+1])
    #     JV[:, j] = 0.5 * (J[:, j] + J[:, j+1])
    #     x_xiV[:, j] = 0.5 * (x_xi[:, j] + x_xi[:, j+1])
    #     x_etaV[:, j] = 0.5 * (x_eta[:, j] + x_eta[:, j+1])
    #     y_xiV[:, j] = 0.5 * (y_xi[:, j] + y_xi[:, j+1])
    #     y_etaV[:, j] = 0.5 * (y_eta[:, j] + y_eta[:, j+1])

    g11V[:, :]    = 0.5 * (g11[:, :-1] + g11[:, 1:])
    g12V[:, :]    = 0.5 * (g12[:, :-1] + g12[:, 1:])
    g22V[:, :]    = 0.5 * (g22[:, :-1] + g22[:, 1:])
    JV[:, :]      = 0.5 * (J[:, :-1] + J[:, 1:])
    x_xiV[:, :]   = 0.5 * (x_xi[:, :-1] + x_xi[:, 1:])
    x_etaV[:, :]  = 0.5 * (x_eta[:, :-1] + x_eta[:, 1:])
    y_xiV[:, :]   = 0.5 * (y_xi[:, :-1] + y_xi[:, 1:])
    y_etaV[:, :]  = 0.5 * (y_eta[:, :-1] + y_eta[:, 1:])

    # malla horizontal
    # for i in range(M-1):
    #     g11H[i, :] = 0.5 * (g11[i, :] + g11[i+1, :])
    #     g12H[i, :] = 0.5 * (g12[i, :] + g12[i+1, :])
    #     g22H[i, :] = 0.5 * (g22[i, :] + g22[i+1, :])
    #     JH[i, :] = 0.5 * (J[i, :] + J[i+1, :])
    #     x_xiH[i, :] = 0.5 * (x_xi[i, :] + x_xi[i+1, :])
    #     x_etaH[i, :] = 0.5 * (x_eta[i, :] + x_eta[i+1, :])
    #     y_xiH[i, :] = 0.5 * (y_xi[i, :] + y_xi[i+1, :])
    #     y_etaH[i, :] = 0.5 * (y_eta[i, :] + y_eta[i+1, :])

    g11H[:, :]    = 0.5 * (g11[:-1, :] + g11[1:, :])
    g12H[:, :]    = 0.5 * (g12[:-1, :] + g12[1:, :])
    g22H[:, :]    = 0.5 * (g22[:-1, :] + g22[1:, :])
    JH[:, :]      = 0.5 * (J[:-1, :] + J[1:, :])
    x_xiH[:, :]   = 0.5 * (x_xi[:-1, :] + x_xi[1:, :])
    x_etaH[:, :]  = 0.5 * (x_eta[:-1, :] + x_eta[1:, :])
    y_xiH[:, :]   = 0.5 * (y_xi[:-1, :] + y_xi[1:, :])
    y_etaH[:, :]  = 0.5 * (y_eta[:-1, :] + y_eta[1:, :])

    # se calcula el ángulo theta de cada nodo, resultado en ángulos absolutos
    # desde 0 hasta 2 * pi
    theta = np.arctan2(Y, X)
    mask = theta < 0
    theta[mask] += 2 * np.pi
    theta[-1, :] = 2 * np.pi
    theta[0, :] = 0

    # convierte angulo de ataque a radianes
    alfa = alfa * np.pi / 180
    C = 0
    phi = np.zeros((M, N))
    UH = np.zeros((M-1, N))
    VH = np.zeros((M-1, N))
    UV = np.zeros((M, N-1))
    VV = np.zeros((M, N-1))
    PH = np.zeros((M-1, N))
    PV = np.zeros((M, N-1))

    err = 1.0
    it_max = 30000
    it = 0
    error = 1e-8
    omega = 1.5 # 1.5
    IMA = 0

    arcotan = np.zeros((M,))
    arcosen = np.zeros((M,))

    arcotan[:] = np.arctan((1 - mach_inf ** 2) ** 0.5
                           * np.tan(theta[:, 0] - alfa))
    arcosen[:] = np.arcsin((1 - mach_inf ** 2) ** 0.5
                           * np.sin(theta[:, 0] - alfa))
    for i in range(M):
        if arcotan[i] > 0 and arcosen[i] < 0:
            arcotan[i] += np.pi
        elif arcotan[i] < 0 and arcosen[i] > 0:
            arcotan[i] = np.pi - np.abs(arcotan[i])
        elif arcotan[i] < 0 and arcosen[i] < 0:
            if theta[i, 0] - alfa > 0:
                arcotan[i] += (2 * np.pi)

    error = 1e-5
    print('Potential Flow')
    while err > error and it < it_max:
        print('it =  ' + str(it), end=' ')
        print('err = ' + '{0:.4e}'.format(err), end=' ')
        print('C = ' + '{:.5e}'.format(C), end=' ')
        print('\t', end='\r')
        it += 1
        phi_old = np.copy(phi)

        # Función potencial en la frontera externa ec 4.18
        phi[:, 0] = v_inf * (X[:, 0] * np.cos(alfa) + Y[:, 0]
                              * np.sin(alfa)) + C * arcotan[:] / 2 / np.pi

        # Nodos internos de la malla
        # velocidades U y V en mallas intercaladas V y H (vertical, horizontal)
        # malla vertical
        for j in range(N-1):
            PV[0, j] = 0.25 * (phi[1, j+1] - phi[-2, j+1]
                                + phi[1, j] - phi[-2, j] + 2 * C)  # 2c?
            for i in range(1, M-1):
                PV[i, j] = 0.25 * (phi[i+1, j+1] - phi[i-1, j+1]
                                   + phi[i+1, j] - phi[i-1, j])

        # quizá se le deba restar la circulación
        PV[-1, :] = PV[0, :]
        # PV[-1, :] = (4 * PV[0, :] - 2 * C) / 4

        for j in range(N-1):
            UV[:, j] = g11V[:, j] * PV[:, j] + g12V[:, j] \
                    * (phi[:, j+1] - phi[:, j])
            VV[:, j] = g12V[:, j] * PV[:, j] + g22V[:, j] \
                 * (phi[:, j+1] - phi[:, j])

        # malla horizontal
        for i in range(M-1):
            PH[i, -1] = 0.25 * (3 * phi[i+1, -1] - 4 * phi[i+1, -2]
                                + phi[i+1, j-3] + 3 * phi[i, -1]
                                - 4 * phi[i, -2] + phi[i, -3])
            PH[i, 0] = 0.25 * (-3 * phi[i+1, 0] + 4 * phi[i+1, 1]
                               - phi[i+1, 2] - 3 * phi[i, 0] + 4 * phi[i, 1]
                               - phi[i, 2])
            for j in range(1, N-2):
                PH[i, j] = 0.25 * (phi[i+1, j+1] - phi[i+1, j-1]
                                   + phi[i, j+1] - phi[i, j-1])
        for i in range(M-1):
            UH[i, 1 : N-1] = g11H[i, 1 : N-1] * (phi[i+1, 1 : N-1] \
                                - phi[i, 1:N-1]) + g12H[i, 1:N-1] * PH[i, 1:N-1]
            VH[i, 1 : N-1] = g12H[i, 1:N-1] * (phi[i+1, 1:N-1] \
                                - phi[i, 1:N-1]) + g22H[i, 1:N-1] * PH[i, 1:N-1]

        rhoV = 1 - ((UV**2 * (x_xiV**2 + y_xiV**2)
                    + VV**2 * (x_etaV**2 + y_etaV**2)
                    + 2 * UV * VV * (x_xiV * x_etaV + y_xiV * y_etaV))\
                    / 2 / H0)
        rhoH = 1 - ((UH**2 * (x_xiH**2 + y_xiH**2)
                    + VH**2 * (x_etaH**2 + y_etaH**2)
                    + 2 * UH * VH * (x_xiH * x_etaH + y_xiH * y_etaH))
                    / 2 / H0)

        g21V = g12V

        # checando los valores de la densidad
        IMA = 0
        mV = rhoV < 0
        mH = rhoH < 0
        if mV.any() or mH.any():
            IMA = 1
        rhoV_tmp = np.copy(rhoV)
        rhoH_tmp = np.copy(rhoH)

        rhoV = d0 * np.abs(rhoV_tmp) ** (1 / (gamma - 1))
        rhoH = d0 * np.abs(rhoH_tmp) ** (1 / (gamma - 1))

        # cálculo de función potencial phi
        for i in range(M-1):
            for j in range(1, N-1):
                if i == 0:
                    phi[i, j] = (rhoH[i, j] * JH[i, j] \
                                    * (g12H[i, j] * PH[i, j] \
                                       + g11H[i, j] * phi[i+1, j]) \
                                    - rhoH[M-2, j] * JH[M-2, j] \
                                        * (g12H[M-2, j] * PH[M-2, j] \
                                        - g11H[M-2, j] * (phi[M-2, j] - C))
                                    + rhoV[i, j] * JV[i, j] \
                                        * (g21V[i, j] * PV[i, j] \
                                        + g22V[i, j] * phi[i, j+1]) \
                                    - rhoV[i, j-1] * JV[i, j-1] \
                                        * (g21V[i, j-1] * PV[i, j-1] \
                                        - g22V[i, j-1] * phi[i, j-1])) \
                                / (rhoH[i, j] * JH[i, j] * g11H[i, j] \
                                   + rhoH[M-2, j] * JH[M-2, j] * g11H[M-2, j] \
                                   + rhoV[i, j] * JV[i, j] * g22V[i, j] \
                                   + rhoV[i, j-1] * JV[i, j-1] * g22V[i, j-1])
                else:
                    phi[i, j] = (rhoH[i, j] * JH[i, j] \
                                    * (g12H[i, j] * PH[i, j] \
                                        + g11H[i,j] * phi[i+1, j]) \
                                    - rhoH[i-1, j] * JH[i-1, j] \
                                        * (g12H[i-1, j] * PH[i-1, j] \
                                            - g11H[i-1, j] * (phi[i-1, j])) \
                                    + rhoV[i, j] * JV[i, j] \
                                        * (g21V[i, j] * PV[i, j] \
                                        + g22V[i, j] * phi[i, j+1]) \
                                    - rhoV[i, j-1] * JV[i, j-1] \
                                        * (g21V[i, j-1] * PV[i, j-1] \
                                        - g22V[i, j-1] * phi[i, j-1])) \
                                / (rhoH[i, j] * JH[i, j] * g11H[i, j] \
                                    + rhoH[i-1, j] * JH[i-1, j] * g11H[i-1, j] \
                                    + rhoV[i, j] * JV[i, j] * g22V[i, j] \
                                    + rhoV[i, j-1] * JV[i, j-1] * g22V[i, j-1])

        # Aplicamos el método SOR de sobrerelajación, ecuación.
        phi = omega * phi + (1 - omega) * phi_old

        # condición en la superficie del perfil
        for i in range(M-2, 0, -1):
            phi[i, N-1] = 1 / 3 * (4 * phi[i, N-2] - phi[i, N-3] - g12[i, N-1]
                            / g22[i, N-1] * (phi[i+1, N-1] - phi[i-1, N-1]))

        phi[0, N-1] = 1 / 3 * (4 * phi[0, N-2] - phi[0, N-3] - g12[0, N-1]
                            / g22[0, N-1] * (phi[1, N-1] - phi[M-2, N-1] + C))

        # discontinuidad del potencial
        phi[M-1, :] = phi[0, :] + C

        err = abs(phi - phi_old).max()

        # cálculo de la Circulación
        C = phi[M-2, N-1] - phi[1, N-1] - g12[0, N-1] * \
            (phi[0, N-3] - 4 * phi[0, N-2] + 3 * phi[0, N-1]) / g11[0, N-1]

    print('\noutside while. it = ' + str(it))
    print('IMA = ' + str(IMA))

    phi = np.flip(phi)
    theta = np.flip(theta)
    return (phi, C, theta, IMA)


def velocity(alfa, C, mach_inf, theta, mesh, phi, v_inf):
    """
    Calcula el vector de velocidad en cada uno de los puntos de la malla
    ...

    Parametros
    ----------
    alfa : float64
        Angulo de ataque
    C : float64
        circulacion alrededor del perfil
    mach_inf : float64
        numero de mach de la corriente libre
    theta : numpy.array
        angulo entre el eje X y todos los nodos de la malla
    mesh : mesh
        Objeto mesh, que contiene toda la informacion relativa a la malla
    phi : numpy.array
        valores de la funcion de potencial en todos los nodos de la malla
    v_inf : float64
        velocidad de la corriente libre

    Return
    ------
    (X, Y) : numpy.array, numpy.array
        Matrices X y Y que describen la malla. Actualizadas.
    """

    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)
    theta = np.flip(theta)
    phi = np.flip(phi)
    M = mesh.M
    N = mesh.N

    u = np.zeros((M, N))
    v = np.zeros((M, N))

    # se obtiene el tensor de la malla
    (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, _, _, _) = \
        mesh.tensor()
    X = np.copy(mesh.X)
    Y = np.copy(mesh.Y)
    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)

    # condición de frontera exterior
    alfa = alfa * np.pi / 180

    for i in range(M):
        j = 0
        # u[i, j] = v_inf * np.cos(alfa) + (C / 2 / np.pi)\
        #             * (1 - mach_inf ** 2) ** 0.5\
        #             * (1 + np.tan(theta[i, j] - alfa) ** 2)\
        #             * (1 / (1 + (Y[i, j] / X[i, j]) ** 2))\
        #             * (-Y[i, j] / (X[i, j]) ** 2)\
        #             / (1 + (1 - mach_inf ** 2)\
        #                 * np.tan(theta[i, j] - alfa) ** 2)
        u[i, j] = v_inf * np.cos(alfa) + (C / 2 / np.pi)\
                    * (1 - mach_inf ** 2) ** 0.5\
                    * (1 + np.tan(theta[i, j] - alfa) ** 2)\
                    * (1 / (1 + (Y[i, j] / X[i, j]) ** 2))\
                    * (-Y[i, j] / X[i, j])\
                    / (1 + (1 - mach_inf ** 2)\
                        * np.tan(theta[i, j] - alfa) ** 2)
        v[i, j] = v_inf * np.sin(alfa) + (C / 2 / np.pi)\
                    * (1 - mach_inf ** 2) ** 0.5\
                    * (1 + np.tan(theta[i, j] - alfa) ** 2)\
                    * (1 / (1 + (Y[i, j] / X[i, j]) ** 2))\
                    * (1 / X[i, j])\
                    / (1 + (1 - mach_inf ** 2)\
                        * np.tan(theta[i, j] - alfa) ** 2)

    # condición nodos interiores de la malla
    for i in range(1, M-1):
        for j in range(1, N-1):
            u[i, j] = (1 / J[i, j]) * (((phi[i+1, j] - phi[i-1, j]) / 2)\
                            * y_eta[i, j] - ((phi[i, j+1] - phi[i, j-1]) / 2)\
                            * y_xi[i, j])
            v[i, j] = (1 / J[i, j]) * (((phi[i, j+1] - phi[i, j-1]) / 2)
                            * x_xi[i, j] - ((phi[i+1, j] - phi[i-1, j]) / 2)
                            * x_eta[i, j])

    for j in range(1, N-1):
        u[0, j] = (1 / J[0, j]) * (((phi[1, j] - phi[-2, j] + C) / 2)\
                            * y_eta[0, j] - ((phi[0, j+1] - phi[0, j-1]) / 2)\
                            * y_xi[0, j])
        v[0, j] = (1 / J[0, j]) * (((phi[0, j+1] - phi[0, j-1]) / 2)\
                            * x_xi[0, j] - ((phi[1, j] - phi[-2, j] + C) / 2)\
                            * x_eta[0, j])

    u[-1, :] = u[0, :]
    v[-1, :] = v[0, :]

    # condición en frontera interior
    j = -1
    for i in range(1, M-1):
        u[i, j] = (((phi[i+1, j] - phi[i-1, j]) / 2) / J[i, j]) * (y_eta[i, j]
                    + y_xi[i, j] * g12[i, j] / g22[i, j])
        v[i, j] = - (((phi[i+1, j] - phi[i-1, j]) / 2) / J[i, j]) * (x_eta[i, j]
                    + x_xi[i, j] * g12[i, j] / g22[i, j])

    u = np.flip(u)
    v = np.flip(v)

    return (u, v)

def pressure(u, v, v_inf, d_inf, gamma, p_inf, p0, d0, h0):
    """
    Calcula el coeficiente de presion en toda la malla
    ...

    Parametros
    ----------
    u, v : numpy.array
        componentes del vector de velocidad
    v_inf : float64
        velocidad de la corriente libre
    d_inf : float64
        densidad de la corriente libre
    gamma : float64
        relacion de calores especificos del fluido
    p_inf : float64
        presion de la corriente libre
    p0 : float64
        presion de estancamiento
    d0 : float64
        densidad de estancamiento
    h0 : float64
        entalpia de estancamiento

    Return
    ------
    (cp, p) : numpy.array, numpy.array
        cp: coeficiente de presion
        p: presion estatica
    """

    # de las ecuaciones de flujo potencial
    d = d0 * (1 - (u ** 2 + v ** 2) / 2 / h0) ** (1 / (gamma - 1))

    # relaciones isentropicas
    p = p0 / (d0 / d) ** gamma
    # p = (p0 - p_inf) * (d / d0) ** gamma
    # p = p0 * (d / d0) ** gamma
    # cp = np.real(2 * (p - p_inf) / (d_inf * v_inf ** 2))
    # cp = 1 - (u ** 2 + v ** 2) / v_inf ** 2
    cp = (p - p_inf) / (p0 - p_inf)

    return (cp, p)

def streamlines(u, v, gamma, h0, d0, p, mesh):
    """
    Resuelve la funcion de corriente
    ...

    Parametros
    ----------
    u, v : numpy.array
        componentes del vector de velocidad
    gamma : float64
        relacion de calores especificos del fluido
    h0 : float64
        entalpia de estancamiento
    d0 : float64
        densidad de estancamiento
    p : float64
        presion de estatica
    mesh : mesh
        Objeto mesh, que contiene toda la informacion relativa a la malla
        sobre la cual se resolvera el flujo potencial

    Return
    ------
    (psi, mach) : numpy.array, numpy.array
        psi: funcion de corriente
        mach: numero de mach
    """

    M = mesh.M
    N = mesh.N
    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)
    u = np.flip(u)
    v = np.flip(v)
    p = np.flip(p)

    psi = np.zeros((M, N))
    JU = np.zeros((M, N))
    c_ = np.zeros((M, N))
    mach = np.zeros((M, N))

    (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, _, _, _) = \
        mesh.tensor()
    X = np.copy(mesh.X)
    Y = np.copy(mesh.Y)
    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)

    d = d0 * (1 - (u ** 2 + v ** 2) / 2 / h0) ** (1 / (gamma - 1))

    JU = u * y_eta - v * x_eta

    for i in range(M):
        for j in range(N-1, 0, -1):
            psi[i, j-1] = psi[i, j] + JU[i, j] * d[i, j]

    c_ = (gamma * p / d) ** 0.5
    mach = (u ** 2 + v ** 2) ** 0.5 / c_

    psi = np.flip(psi)
    mach = np.flip(mach)
    u = np.flip(u)
    v = np.flip(v)

    return (psi, mach)

def lift_n_drag(mesh, cp, alfa, c):
    """
    Calculo de coeficientes de sustentacion y resistencia aerodinamica
    ...

    Parametros
    ----------
    mesh : mesh
        Objeto mesh, que contiene toda la informacion relativa a la malla
        sobre la cual se resolvera el flujo potencial
    cp : numpy.array
        coeficiente de presion
    alfa : float64
        angulo de ataque
    c : float64
        cuerda aerodinamica del perfil

    Return
    ------
    (L, D) : float64, float64
        L: coeficiente de levantamiento
        D: coeficiente de resistencia aerodinamica
    """

    M = mesh.M
    N = mesh.N
    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)
    cp = np.flip(cp)
    (_, _, _, _, x_xi, _, y_xi, _, _, _, _) = \
        mesh.tensor()
    mesh.X = np.flip(mesh.X)
    mesh.Y = np.flip(mesh.Y)

    alfa = alfa * np.pi / 180

    kx = np.zeros((M,))
    ky = np.zeros((M,))

    kx = cp[:, -1] * y_xi[:, -1] / c
    ky = cp[:, -1] * x_xi[:, -1] / c

    xi = np.linspace(1, M, M)
    cl_x = -np.trapz(kx, xi)
    cl_y = np.trapz(ky, xi)

    L = -cl_x * np.sin(alfa) + cl_y * np.cos(alfa)
    D = cl_x * np.cos(alfa) + cl_y * np.sin(alfa)

    return (L, D)
