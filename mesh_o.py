#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 1 13:53:21 2018

@author: cardoso

Define subclase mesh_O.
Se definen diversos métodos de generación para este tipo de mallas
"""

from mesh import mesh
import numpy as np


class mesh_O(mesh):
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        '''
        mesh.__init__(self, R, M, N, archivo)
        self.tipo = 'O'
        self.fronteras()

    def fronteras(self):
        '''
        Genera la frontera externa de la malla así como la interna
        '''
        R = self.R
        # cargar datos del perfil
        perfil = np.loadtxt(self.archivo)
        perfil_x = perfil[:, 0]
        perfil_y = perfil[:, 1]
        points = np.shape(perfil_x)[0]
        points = (points + 1) // 2

        # frontera externa
        theta = np.linspace(0, np.pi, points)
        theta2 = np.linspace(np.pi, 2 * np.pi, points)
        theta = np.concatenate((theta, theta2[1:]))
        del theta2
        x = R * np.cos(theta)
        y = R * np.sin(theta)

        x = np.flip(x, 0)
        y = np.flip(y, 0)
        # primera columna FI (perfil), ultima columna FE
        self.X[:, -1] = x
        self.Y[:, -1] = y
        self.X[:, 0] = perfil_x
        self.Y[:, 0] = perfil_y
        return

    # funcion para generar mallas mediante  ecuación de Laplace.
    def gen_Laplace(self, metodo='SOR'):
        '''
        Genera malla resolviendo ecuación de Laplace
        metodo = J (Jacobi), GS (Gauss-Seidel), SOR (Sobre-relajacion)
        '''

        # se genera malla antes por algún método algebráico
        self.gen_TFI()

        # se inician variables
        Xn = self.X
        Yn = self.Y
        m = self.M
        n = self.N

        d_eta = self.d_eta
        d_xi = self.d_xi
        omega = 1.4  # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. Solución tiende a diverger
            omega = 1     ---> método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. acelera la convergencia.
        '''

        it = 0
        print("Laplace:")

        # inicio del método iterativo
        while it < mesh.it_max:
            print(it, end='\r')
            Xo = np.copy(Xn)
            Yo = np.copy(Yn)

            # si el método iterativo es Jacobi
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:   # si el método es Gauss-Seidel o SOR
                X = Xn
                Y = Yn

            for j in range(1, n-1):
                for i in range(1, m-1):
                    x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                    y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                    x_xi = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
                    y_xi = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

                    alpha = x_eta ** 2 + y_eta ** 2
                    beta = x_xi * x_eta + y_xi * y_eta
                    gamma = x_xi ** 2 + y_xi ** 2

                    Xn[i, j] = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                        * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                            + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                                    - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1]))

                    Yn[i, j] = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                        * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                            + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                                    - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1]))

                i = m-1
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[1, j] - X[i-1, j]) / 2 / d_xi
                y_xi = (Y[1, j] - Y[i-1, j]) / 2 / d_xi
                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2
                Xn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                    * (alpha / (d_xi**2) * (X[1, j] + X[i-1, j])
                        + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])
                        - beta / (2 * d_xi * d_eta) * (X[1, j+1] - X[1, j-1]
                                                + X[i-1, j-1] - X[i-1, j+1]))
            Xn[0, :] = Xn[-1, :]
            Yn[0, :] = Yn[-1, :]

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            it += 1

            if abs(Xn - Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print(metodo + ': saliendo...')
                print('it=', it)
                break

        self.X = Xn
        self.Y = Yn
        return

    def gen_Poisson(self, metodo='SOR'):
        '''
        Genera malla resolviendo ecuación de Laplace
        metodo = J (Jacobi), GS (Gauss-Seidel), SOR (Sobre-relajacion)
        '''

        # se genera malla antes por algún método algebráico
        self.gen_TFI()

        # se inician variables
        Xn = self.X
        Yn = self.Y
        m = self.M
        n = self.N

        d_eta = self.d_eta
        d_xi = self.d_xi
        omega = np.longdouble(1.5)  # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. Solución tiende a diverger
            omega = 1     ---> método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. acelera la convergencia.
        '''

        # parámetros de ecuación de Poisson
        Q = 0
        P = 0
        I = 0
        a = np.longdouble(0)
        c = np.longdouble(0)
        aa = np.longdouble(0.7)
        cc = np.longdouble(6.5)
        linea_eta = 0.0
        linea_xi = 0.0

        it = 0
        print("Poisson:")
        # inicio del método iterativo
        while it < mesh.it_max:
            print(it, end='\r')
            Xo = np.copy(Xn)
            Yo = np.copy(Yn)
            # si el método iterativo es Jacobi
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:   # si el método es Gauss-Seidel o SOR
                X = Xn
                Y = Yn
            for j in range(1, n-1):
                for i in range(1, m-1):
                    x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                    y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                    x_xi = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
                    y_xi = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

                    alpha = x_eta ** 2 + y_eta ** 2
                    beta = x_xi * x_eta + y_xi * y_eta
                    gamma = x_xi ** 2 + y_xi ** 2

                    if np.abs(i / (m-1) - linea_xi) == 0:
                        P = np.longdouble(0)
                    else:
                        P = -a * (np.longdouble(i / (m-1) - linea_xi))\
                                / np.abs(np.longdouble(i / (m-1) - linea_xi))\
                                * np.exp(-c
                                * np.abs(np.longdouble(i / (m-1) - linea_xi)))

                    if np.abs(j / (n-1) - linea_eta) == 0:
                        Q = 0
                    else:
                        Q = -aa * (np.longdouble(j / (n-1) - linea_eta))\
                                / np.abs(np.longdouble(j / (n-1) - linea_eta))\
                                * np.exp(-cc
                                * np.abs(np.longdouble(j / (n-1) - linea_eta)))
                    I = x_xi * y_eta - x_eta * y_xi

                    Xn[i, j] = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                        * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                            + gamma / (d_eta ** 2) * (X[i, j+1] + X[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                                    - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])
                            + I ** 2 * (P * x_xi + Q * x_eta))
                    Yn[i, j] = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                        * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                            + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                                    - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])
                            + I**2 * (P * y_xi + Q * y_eta))

                i = m-1
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[1, j] - X[i-1, j]) / 2 / d_xi
                y_xi = (Y[1, j] - Y[i-1, j]) / 2 / d_xi

                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2

                if np.abs(i / (m-1) - linea_xi) == 0:
                    P = 0
                else:
                    P = -a * (i / (m-1) - linea_xi)\
                            / np.abs(i / (m-1) - linea_xi)\
                            * np.exp(-c * np.abs(i / (m-1) - linea_xi))

                if np.abs(j / (n-1) - linea_eta) == 0:
                    Q = 0
                else:
                    Q = -aa * (np.longdouble(j / (n-1) - linea_eta))\
                            / np.abs(np.longdouble(j / (n-1) - linea_eta))\
                            * np.exp(-cc
                                * np.abs(np.longdouble(j / (n-1) - linea_eta)))
                I = x_xi * y_eta - x_eta * y_xi

                Xn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                    * (alpha / (d_xi**2) * (X[1, j] + X[i-1, j])
                        + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])
                        - beta / (2 * d_xi * d_eta)
                        * (X[1, j+1] - X[1, j-1] + X[i-1, j-1] - X[i-1, j+1])
                        + I**2 * (P * x_xi + Q * x_eta))
            Xn[0, :] = Xn[-1, :]
            Yn[0, :] = Yn[-1, :]

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            it += 1

            if abs(Xn - Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print(metodo + ': saliendo...')
                print('it = ', it)
                break

        self.X = Xn
        self.Y = Yn
        return

    def gen_hyperbolic(self):
        '''
        Genera mallas hiperbólicas. Método de Steger
        '''
        # se inician las variables características de la malla
        m = self.M
        n = self.N
        X = self.X
        Y = self.Y
        d_xi = self.d_xi
        d_eta = self.d_eta
        d_s1 = 0.01
        S = np.zeros((m - 2, m - 2), dtype=object)
        L = np.zeros((m - 2, m - 2), dtype=object)
        U = np.zeros((m - 2, m - 2), dtype=object)
        R = np.zeros((m - 2, 1), dtype=object)
        Z = np.zeros((m - 2, 1), dtype=object)
        DD = np.zeros((m - 2, 1), dtype=object)
        Fprev = 0.05
        C = np.zeros((2, 2))
        self.gen_TFI()
        for j in range(1, n):
            # se llena la matriz S y el vector DD
            for i in range(1, m-1):
                F = 0.5 * (((X[i, j - 1] - X[i - 1, j - 1]) ** 2
                            + (Y[i, j - 1] - Y[i - 1, j - 1]) ** 2) ** 0.5
                           + ((X[i + 1, j - 1] - X[i, j - 1]) ** 2
                              + (Y[i + 1, j - 1] - Y[i, j - 1]) ** 2) ** 0.5)
                F = F * d_s1 * (1 + 0.05) ** (j - 1)
                x_xi_k = (X[i + 1, j - 1] - X[i - 1, j - 1]) / 2 / d_xi
                y_xi_k = (Y[i + 1, j - 1] - Y[i - 1, j - 1]) / 2 / d_xi
                x_eta_k = - y_xi_k * F / (x_xi_k ** 2 + y_xi_k ** 2)
                y_eta_k = x_xi_k * F / (x_xi_k ** 2 + y_xi_k ** 2)
                B_1 = np.array([[x_xi_k, -y_xi_k], [y_xi_k, x_xi_k]])\
                    / (x_xi_k ** 2 + y_xi_k ** 2)
                A = np.array([[x_eta_k, y_eta_k], [y_eta_k, -x_eta_k]])
                C = B_1 @ A
                AA = - 1 / 2 / d_xi * C
                BB = np.identity(2) / d_eta
                CC = -AA
                dd = B_1 @ np.array([[0], [F + Fprev]])\
                    + np.array([[X[i, j - 1]], [Y[i, j - 1]]]) / d_eta
                if i == 1:
                    dd -= (AA @ np.array([[X[0, j]], [Y[0, j]]]))
                    S[0, 0] = BB
                    S[0, 1] = CC
                elif i == m - 2:
                    dd -= (CC @ np.array([[X[m - 1, j]], [Y[m - 1, j]]]))
                    S[m - 3, m - 4] = AA
                    S[m - 3, m - 3] = BB
                else:
                    S[i - 1, i - 2] = AA
                    S[i - 1, i - 1] = BB
                    S[i - 1, i] = CC
                DD[i - 1, 0] = dd
            # se llenan las matrices L y U
            for i in range(m - 2):
                if i == 0:
                    L[0, 0] = S[0, 0]
                    U[0, 0] = np.identity(2)
                    U[0, 1] = np.linalg.inv(S[0, 0]) @ S[0, 1]
                elif i == m - 3:
                    L[m - 3, m - 4] = S[m - 3, m - 4]
                    L[m - 3, m - 3] = S[m - 3, m - 3]\
                        - S[m - 3, m - 4] @ U[m - 4, m - 3]
                    U[m - 3, m - 3] = np.identity(2)
                else:
                    L[i, i - 1] = S[i, i - 1]
                    L[i, i] = S[i, i] - S[i, i - 1] @ U[i - 1, i]
                    U[i, i] = np.identity(2)
                    U[i, i + 1] = np.linalg.inv(L[i, i]) @ S[i, i + 1]
            # se obtienen los valores del vector Z
            i = 0
            Z[0, 0] = np.linalg.inv(L[0, 0]) @ DD[0, 0]
            for i in range(1, m - 2):
                Z[i, 0] = np.linalg.inv(L[i, i])\
                        @ (DD[i, 0] - L[i, i - 1] @ Z[i - 1, 0])
            # se obtienen los valores del vector R
            i = m - 3
            R[i, 0] = Z[i, 0]
            for i in range(m - 4, -1, -1):
                R[i, 0] = Z[i, 0] - U[i, i + 1] @ Z[i + 1, 0]
            # se asignan las coordenadas X y Y
            for i in range(1, m - 1):
                X[i, j] = R[i - 1, 0][0]
                Y[i, j] = R[i - 1, 0][1]
            x_xi = (X[1, j - 1] - X[-2, j - 1]) / 2 / d_xi
            y_xi = (Y[1, j - 1] - Y[-2, j - 1]) / 2 / d_xi
            X[0, j] = X[0, j - 1] - d_eta / 2 / d_xi * F\
                / (x_xi ** 2 + y_xi ** 2)
            X[0, j] *= (Y[1, j - 1] - Y[-2, j - 1])
            X[-1, j] = X[0, j]
            Fprev = F
        return
