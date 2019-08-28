#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 26 2019

@author: cardoso

Códigos para la transformación de la metrica
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def tensor_o(mesh):
    '''
        Calcula el tensor métrico de la transformación para ambas
            transformaciones, directa e indirecta
        Calcula el Jacobiano de la matriz de transformación
        Calcula el valor discretizado de las derivadas parciales:
            x_xi
            x_eta
            y_xi
            y_eta
    '''

    # se definen vairables de la malla
    X = mesh.X
    Y = mesh.Y
    M = mesh.M
    N = mesh.N
    d_xi = mesh.d_xi
    d_eta = mesh.d_eta

    # se crean matrices
    x_xi = np.zeros((M, N))
    x_eta = np.zeros((M, N))
    y_xi = np.zeros((M, N))
    y_eta = np.zeros((M, N))

    # cálculo de derivadas parciales
    # nodos internos
    for j in range(1, N-1):
        x_eta[:-1, j] = (X[:-1, j+1] - X[:-1, j-1]) / 2 / d_eta
        y_eta[:-1, j] = (Y[:-1, j+1] - Y[:-1, j-1]) / 2 / d_eta
    x_eta[:-1, 0] = (X[:-1, 1] - X[:-1, 0]) / d_eta
    x_eta[:-1, -1] = (X[:-1, -1] - X[:-1, -2]) / d_eta
    x_eta[-1, :] = x_eta[0, :]
    y_eta[:-1, 0] = (Y[:-1, 1] - Y[:-1, 0]) / d_eta
    y_eta[:-1, -1] = (Y[:-1, -1] - Y[:-1, -2]) / d_eta
    y_eta[-1, :] = y_eta[0, :]

    for i in range(0, M-1):
        x_xi[i, :] = (X[i+1, :] - X[i-1, :]) / 2 / d_xi
        y_xi[i, :] = (Y[i+1, :] - Y[i-1, :]) / 2 / d_xi
    x_xi[0, :] = (X[1, :] - X[-2, :]) / 2 / d_xi
    y_xi[0, :] = (Y[1, :] - Y[-2, :]) / 2 / d_xi
    x_xi[-1, :] = x_xi[0, :]
    y_xi[-1, :] = y_xi[0, :]

    # obteniendo los tensores de la métrica
    J = (x_xi * y_eta) - (x_eta * y_xi)
    g11I = x_xi ** 2 + y_xi ** 2
    g12I = x_xi * x_eta + y_xi * y_eta
    g22I = d_eta ** 2 + y_eta ** 2
    g11 = g22I / J ** 2
    g12 = -g12I / J ** 2
    g22 = g11I / J ** 2

    C1 = g11I
    A = g22I
    B = g12I

    return (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, A, B, C1)


def potential_flow(d0, H0, gamma, mach_inf, v_inf, alfa, mesh):
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

    # calculo del tensor de la métrica
    (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, A, B, C1) = mesh.tensor()

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

    # se calcula el ángulo theta de cada nodo, resultado en ángulos absolutos
    # desde 0 hasta 2 * pi
    theta = np.arctan2(Y, X)
    mask = theta < 0
    theta[mask] += 2 * np.pi
    # convierte angulo de ataque a radianes
    alfa = alfa * np.pi / 180

    ###########################################################################
    #
    #   Inicia proceso iterativo para la solución del flujo potencial
    #
    ###########################################################################
    C = 0.5
    phi = np.zeros((M, N))
    UH = np.zeros((M, N))
    VH = np.zeros((M, N))
    UV = np.zeros((M, N))
    VV = np.zeros((M, N))
    ddd = 1
    it_max = 20000
    it = 0
    error = 1e-9
    omega = 0.5

    arcotan = np.zeros((M,))
    arcosen = np.zeros((M,))

    while ddd > error and it < it_max:
        print(it, end='\r')
        it += 1
        phi_old = np.copy(phi)

        # Función potencial en la frontera externa
        arcotan[:] = np.arctan((1 - mach_inf ** 2) ** 0.5
                               * np.tan(theta[:, -1] - alfa))
        arcosen[:] = np.arcsin((1 - mach_inf ** 2) ** 0.5
                               * np.sin(theta[:, -1] - alfa))
        for i in range(M):
            if arcotan[i] > 0 and arcosen[i] < 0:
                arcotan[i] += np.pi
            elif arcotan[i] < 0 and arcosen[i] > 0:
                arcotan[i] = np.pi - abs(arcotan[i])
            elif arcotan[i] < 0 and arcosen[i] < 0:
                if theta[i, -1] - alfa > 0:
                    arcotan[i] += (2 * np.pi)

        phi[:, -1] = v_inf * (X[:, -1] * np.cos(alfa) + Y[:, -1]
                              * np.sin(alfa)) + C * arcotan[:] / 2 / np.pi

        # Nodos internos de la malla
        # velocidades U y V en mallas intercaladas V y H (vertical, horizontal)
        break
