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
from sys import maxsize
from numba import jitclass
from numba import float32, int8, int16, boolean

np.set_printoptions(threshold=maxsize)

class mesh(object):
    it_max = 8000
    err_max = 1e-6

    def __init__(self, R, M, N, airfoil):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del
        perfil se asigna ese valor desde el sript main.py
        airfoil = perfil a analizar
        M = numero de nodos en eje xi
        N = numero de nodos en eje eta
        airfoil_alone = boolean. Si FI es perfil solo (True) o con flap (False)
        airfoil_boundary = np.array tomar un calor si el punto es parte de
            un perfil y 0 (cero) si no es frontera (hay flujo)
        X = matriz cuadrada que contiene todos las coordenadas 'x' de los
            puntos de la malla
        Y = matriz cuadrada que contiene todos las coordenadas 'y' de los
            puntos de la malla
        '''

        self.tipo               = None
        self.d_eta              = 1
        self.d_xi               = 1
        self.R                  = R
        self.M                  = M
        self.N                  = N
        self.airfoil_alone      = airfoil.alone
        self.airfoil_join       = airfoil.union
        self.airfoil_boundary   = airfoil.is_boundary

        self.X                  = np.zeros((M, N))
        self.Y                  = np.zeros((M, N))

        return

    def get_tipo(self):
        return self.tipo

    def get_d_eta(self):
        return self.d_eta

    def get_d_xi(self):
        return self.dxi

    def get_R(self):
        return self.R

    def get_M(self):
        return self.M

    def get_N(self):
        return self.N

    def is_airfoil_alone(self):
        return self.airfoil_alone

    def get_airfoil_join(self):
        return self.arifoil_join

    def get_airfoil_boundary(self):
        return self.airfoil_boundary

    def get_X(self):
        return (np.copy(self.X))

    def get_Y(self):
        return (np.copy(self.Y))

    def plot(self):
        '''
        función para graficar la malla
        '''

        plt.axis('equal')
        plt.plot(self.X, self.Y, 'k', linewidth=0.5)
        plt.plot(self.X[:, 0], self.Y[:, 0], 'k', linewidth=1.9)

        for i in range(self.M):
            plt.plot(self.X[i, :], self.Y[i, :], 'b', linewidth=0.5)
        plt.draw()
        plt.show()

        return

    def to_txt_mesh(self, filename='./garbage/mesh_own.txt_mesh'):
        '''
        Exporta mallas a archivo en formato propio.
        Con el proposito de ser importadas posteriormente y que el porgrama
            reconozca todas sus características
        '''

        file = open(filename, 'w+')

        # se escriben atributos de la malla
        file.write('tipo= ' + str(self.tipo) + ' \n')
        file.write('d_eta= ' + str(self.d_eta) + ' \n')
        file.write('d_xi= ' + str(self.d_xi) + ' \n')
        file.write('R= ' + str(self.R) + ' \n')
        file.write('M= ' + str(self.M) + ' \n')
        file.write('N= ' + str(self.N) + ' \n')
        file.write('airfoil_alone= ' + str(self.airfoil_alone) + ' \n')
        file.write('airfoil_join= ' + str(self.airfoil_join) + ' \n')

        # se escribe array que indica que puntos son fronter y cuales no
        file.write('airfoil_boundary=\n')
        s = ','.join([str(x_) for x_ in self.airfoil_boundary])
        # file.write(np.array2string(self.airfoil_boundary, separator=','))
        file.write(s)
        file.write('\n')

        # se escrriben matriz X y matriz Y
        file.write('X=\n')
        for i in range(self.M):
            s = ','.join([str(x_) for x_ in self.X[i, :]])
            file.write(s + '\n')

        file.write('Y=\n')
        for i in range(self.M):
            s = ','.join([str(y_) for y_ in self.Y[i, :]])
            file.write(s + '\n')

        return

    def gen_inter_pol(self, eje='eta'):
        '''
        genera malla por interpolación polinomial por Lagrange
        sec 4.2.1 M Farrashkhalvat Grid generation
        '''

        Xn = self.X
        Yn = self.Y

        if eje == 'eta':
            n   = self.N
            eta = np.linspace(0, 1, n)

            for j in range(1, n-1):
                Xn[:, j] = Xn[:, 0] * (1 - eta[j]) + Xn[:, -1] * eta[j]
                Yn[:, j] = Yn[:, 0] * (1 - eta[j]) + Yn[:, -1] * eta[j]
            self.X = Xn
            self.Y = Yn
            return (Xn, Yn)

        elif eje == 'xi':
            m   = self.M
            xi  = np.linspace(0, 1, m)
            for i in range(1, m-1):
                Xn[i, :] = Xn[0, :] * (1 - xi[i]) + Xn[-1, :] * xi[i]
                Yn[i, :] = Yn[0, :] * (1 - xi[i]) + Yn[-1, :] * xi[i]
        return

    def gen_TFI(self):
        '''
        genera malla por TFI
        sec 4.3.2 M Farrashkhalvat Grid generation
        '''
        Xn  = self.X
        Yn  = self.Y
        n   = self.N
        eta = np.linspace(0, 1, n)
        m   = self.M
        xi  = np.linspace(0, 1, m)

        for j in range(1, n-1):
            Xn[0, j]    = Xn[0, 0] * (1 - eta[j]) + Xn[0, -1] * eta[j]
            Xn[-1, j]   = Xn[-1, 0] * (1 - eta[j]) + Xn[-1, -1] * eta[j]
            Yn[0, j]    = Yn[0, 0] * (1 - eta[j]) + Yn[0, -1] * eta[j]
            Yn[-1, j]   = Yn[-1, 0] * (1 - eta[j]) + Yn[-1, -1] * eta[j]

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
        Xn      = self.X
        Yn      = self.Y
        n       = self.N
        eta     = np.linspace(0, 1, n)

        derX    = (Xn[:, -1] - Xn[:, 0]) / 1
        derY    = (Yn[:, -1] - Yn[:, 0]) / 200000000
        derX    = np.transpose(derX)
        derY    = np.transpose(derY)
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

    def get_aspect_ratio(self):
        '''
        Calcula el aspect ratio de cada celda. Para cualquier tipo de malla
        Basado en el método de:
            The Verdict Geometric Quality Library
        '''
        aspect_ratio_ = np.zeros((self.M - 1, self.N - 1))

        # se calculan longitudes de los elementos de la celda
        for i in range(0, self.M - 1):
            for j in range(0, self.N -1):
                # dist = (l0, l1, l2, l3)
                dist = (((self.X[i, j] - self.X[i+1, j]) ** 2
                                + (self.Y[i, j] - self.Y[i+1, j]) ** 2) ** 0.5,
                        ((self.X[i+1, j] - self.X[i+1, j+1]) ** 2
                                + (self.Y[i+1, j] - self.Y[i+1, j+1]) ** 2) ** 0.5,
                        ((self.X[i+1, j+1] - self.X[i, j+1]) ** 2
                                + (self.Y[i+1, j+1] - self.Y[i, j+1]) ** 2) ** 0.5,
                        ((self.X[i, j+1] - self.X[i, j]) ** 2
                                + (self.Y[i, j+1] - self.Y[i, j]) ** 2) ** 0.5)

                max_dist = max(dist)

                dist_sum = dist[0] + dist[1] + dist[2] + dist[3]

                # cálculo de productos cruz
                l0_l1 = np.abs((self.X[i+1, j] - self.X[i, j])
                                    *  (self.Y[i+1, j+1] - self.Y[i+1, j])
                                - (self.Y[i+1, j] - self.Y[i, j])
                                     *  (self.X[i+1, j+1] - self.X[i+1, j]))

                l2_l3 = np.abs((self.X[i, j+1] - self.X[i+1, j+1])
                                    * (self.Y[i, j] - self.Y[i, j+1])
                                - (self.Y[i, j+1] - self.Y[i+1, j+1])
                                    * (self.X[i, j] - self.X[i, j+1]))
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
        plt.plot(self.X, self.Y, 'k', linewidth=0.5)
        plt.plot(self.X[:, 0], self.Y[:, 0], 'k', linewidth=0.5)
        for i in range(self.M):
            plt.plot(self.X[i, :], self.Y[i, :], 'k', linewidth=0.5)
        mesh_ = plt.pcolormesh(self.X, self.Y, aspect_ratio_, cmap='jet',
                               rasterized=True, vmin=(aspect_min),
                               vmax=(aspect_max))
        plt.colorbar(mesh_, extend='both')

        plt.draw()
        plt.show()

        return (aspect_ratio_)

    def get_skew(self):
        '''
        Calcula el aspect ratio de cada celda. Para cualquier tipo de malla
        Basado en el método de:
            The Verdict Geometric Quality Library
        '''

        X = np.copy(self.X)
        Y = np.copy(self.Y)
        skew = np.zeros((self.M - 1, self.N - 1))

        for i in range(self.M -1):
            for j in range(self.N -1):
                P0 = np.array(([X[i, j], Y[i, j]]))
                P1 = np.array(([X[i+1, j], Y[i+1, j]]))
                P2 = np.array(([X[i+1, j+1], Y[i+1, j+1]]))
                P3 = np.array(([X[i, j+1], Y[i, j+1]]))
                X1 = (P1 - P0) + (P2 - P3)
                X2 = (P2 - P1) + (P3 - P0)
                mag = (X1[0] ** 2 + X1[1] ** 2) ** 0.5
                X1 /= mag
                mag = (X2[0] ** 2 + X2[1] ** 2) ** 0.5
                X2 /= mag
                skew[i, j] = 1 - np.abs(X1[0] * X2[0] + X1[1] * X2[1])

        skew_min = np.nanmin(skew)
        skew_max = np.nanmax(skew)
        # cmap_ = cm.get_cmap('jet')

        print('skew_max')
        print(skew_max)
        print('skew_min')
        print(skew_min)
        plt.figure('aspect')
        plt.axis('equal')
        plt.plot(self.X, self.Y, 'k', linewidth=0.5)
        plt.plot(self.X[:, 0], self.Y[:, 0], 'k', linewidth=0.5)
        for i in range(self.M):
            plt.plot(self.X[i, :], self.Y[i, :], 'k', linewidth=0.5)
        mesh_ = plt.pcolormesh(self.X, self.Y, skew, cmap='jet', rasterized=True,
                       vmin=(skew_min),
                       vmax=(skew_max))
        plt.colorbar(mesh_)

        plt.draw()
        plt.show()

        return (skew)
