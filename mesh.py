#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:53:21 2018

@author: Cardoso

Define clase mesh. Se generan diferentes subclases para la generación
de diferentes tipos de malla.
Se definen procedimientos para los diferentes métodos de generación de las
mismas.
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


# clase para la generación de mallas
class mesh(object):

    # variables de clase, controlan el numero de iteraciones máximo
    # así como el error maximo permisible como criterio de convergencia
    it_max = 8000
    err_max = 1e-6  #1e-6

    # método de inicialización de instancias de clase
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del
        perfil se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        X = matriz cuadrada que contiene todos las coordenadas 'x' de los
            puntos de la malla
        Y = matriz cuadrada que contiene todos las coordenadas 'y' de los
            puntos de la malla
        '''
        self.R = np.longdouble(R)
        self.M = M
        self.N = N
        self.archivo = archivo

        self.X = np.zeros((M, N))
        self.Y = np.copy(self.X)
        self.d_xi = np.longdouble(1 / (self.M - 1))
        self.d_eta = np.longdouble(1 / (self.N - 1))
        self.tipo = None

    # función para graficar la malla
    def plot(self):
        plt.axis('equal')
        plt.plot(self.X, self.Y, 'k', linewidth=1.8)
        plt.plot(self.X[:, 0], self.Y[:, 0], 'r', linewidth=1.9)
        for i in range(self.M):
            plt.plot(self.X[i, :], self.Y[i, :], 'b', linewidth=0.8)
        plt.show()

    # genera malla por interpolación polinomial por Lagrange
    # sec 4.2.1 M Farrashkhalvat Grid generation
    def gen_inter_pol(self, eje='eta'):
        Xn = self.X
        Yn = self.Y

        if eje == 'eta':
            n = self.N
            eta = np.linspace(0, 1, n)
            for j in range(1, n-1):
                Xn[:, j] = Xn[:, 0] * (1 - eta[j]) + Xn[:, -1] * eta[j]
                Yn[:, j] = Yn[:, 0] * (1 - eta[j]) + Yn[:, -1] * eta[j]
            self.X = Xn
            self.Y = Yn
            return (Xn, Yn)
        elif eje == 'xi':
            m = self.M
            xi = np.linspace(0, 1, m)
            for i in range(1, m-1):
                Xn[i, :] = Xn[0, :] * (1 - xi[i]) + Xn[-1, :] * xi[i]
                Yn[i, :] = Yn[0, :] * (1 - xi[i]) + Yn[-1, :] * xi[i]
        return

    # genera malla por TFI
    # sec 4.3.2 M Farrashkhalvat Grid generation
    def gen_TFI(self):
        Xn = self.X
        Yn = self.Y
        n = self.N
        eta = np.linspace(0, 1, n)
        m = self.M
        xi = np.linspace(0, 1, m)

        for j in range(1, n-1):
            Xn[0, j] = Xn[0, 0] * (1 - eta[j]) + Xn[0, -1] * eta[j]
            Xn[-1, j] = Xn[-1, 0] * (1 - eta[j]) + Xn[-1, -1] * eta[j]
            Yn[0, j] = Yn[0, 0] * (1 - eta[j]) + Yn[0, -1] * eta[j]
            Yn[-1, j] = Yn[-1, 0] * (1 - eta[j]) + Yn[-1, -1] * eta[j]

        for j in range(1, n-1):
            for i in range(1, m-1):
                Xn[i, j] = (1 - xi[i]) * Xn[0, j] + xi[i] * Xn[-1, j] \
                    + (1 - eta[j]) * Xn[i, 0] + eta[j] * Xn[i, -1] \
                    - (1 - xi[i]) * (1 - eta[j]) * Xn[0, 0] \
                    - (1 - xi[i]) * eta[j] * Xn[0, -1] \
                    - (1 - eta[j]) * xi[i] * Xn[-1, 0] \
                    - xi[i] * eta[j] * Xn[-1, -1]

                Yn[i, j] = (1 - xi[i]) * Yn[0, j] + xi[i] * Yn[-1, j] \
                    + (1 - eta[j]) * Yn[i, 0] + eta[j] * Yn[i, -1] \
                    - (1 - xi[i]) * (1 - eta[j]) * Yn[0, 0] \
                    - (1 - xi[i]) * eta[j] * Yn[0, -1] \
                    - (1 - eta[j]) * xi[i] * Yn[-1, 0] \
                    - xi[i] * eta[j] * Yn[-1, -1]

        return

    # genera malla por interpolación de Hermite
    # sec 4.2.2 M Farrashkhalvat Grid generation
    def gen_inter_Hermite(self):
        Xn = self.X
        Yn = self.Y
        n = self.N
        eta = np.linspace(0, 1, n)

        derX = (Xn[:, -1] - Xn[:, 0]) / 1
        derY = (Yn[:, -1] - Yn[:, 0]) / 200000000
        derX = np.transpose(derX)
        derY = np.transpose(derY)
        # Interpolación de hermite
        for j in range(1, n-1):
            Xn[:, j] = Xn[:, 0] * (2 * eta[j]**3 - 3 * eta[j]**2 + 1) \
                + Xn[:, -1] * (3 * eta[j]**2 - 2 * eta[j]**3) \
                + derX * (eta[j] ** 3 - 2 * eta[j]**2 + eta[j]) \
                + derX * (eta[j]**3 - eta[j]**2)
            Yn[:, j] = Yn[:, 0] * (2 * eta[j]**3 - 3 * eta[j]**2 + 1) \
                + Yn[:, -1] * (3 * eta[j]**2 - 2 * eta[j]**3) \
                + derY * (eta[j] ** 3 - 2 * eta[j]**2 + eta[j]) \
                + derY * (eta[j]**3 - eta[j]**2)

        # self.X = Xn
        # self.Y = Yn
        return (Xn, Yn)






