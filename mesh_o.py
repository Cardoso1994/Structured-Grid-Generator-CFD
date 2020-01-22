#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 1 13:53:21 2018

@author: cardoso

Define subclase mesh_O.
Se definen diversos métodos de generación para este tipo de mallas
"""

import numpy as np
import matplotlib.pyplot as plt

from mesh import mesh
import mesh_su2
import sys

np.set_printoptions(threshold=np.sys.maxsize)

class mesh_O(mesh):
    #def __init__(self, R, M, N, airfoil):
    def __init__(self, R, N, airfoil):
        '''
        R = radio de la frontera externa, en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        airfoil = perfil a analizar
        '''
        M = np.shape(airfoil.x)[0]
        mesh.__init__(self, R, M, N, airfoil)
        self.tipo = 'O'
        self.fronteras(airfoil)

    def fronteras(self, airfoil):
        '''
        Genera la frontera externa de la malla así como la interna
        '''
        R = self.R

        # cargar datos del perfil
        perfil      = airfoil
        perfil_x    = perfil.x
        perfil_y    = perfil.y
        points      = np.shape(perfil_x)[0]
        points      = (points + 1) // 2

        # frontera externa
        theta   = np.linspace(0, np.pi, points)
        theta2  = np.linspace(np.pi, 2 * np.pi, points)
        theta   = np.concatenate((theta, theta2[1:]))
        del theta2

        x = R * np.cos(theta)
        y = R * np.sin(theta)

        x = np.flip(x, 0)
        y = np.flip(y, 0)

        # primera columna FI (perfil), ultima columna FE
        self.X[:, -1]   = x
        self.Y[:, -1]   = y
        self.X[:, 0]    = perfil_x
        self.Y[:, 0]    = perfil_y

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
        Xn  = self.X
        Yn  = self.Y
        m   = self.M
        n   = self.N

        d_eta   = self.d_eta
        d_xi    = self.d_xi
        omega   = 1.7  # en caso de metodo SOR
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
                    x_eta   = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                    y_eta   = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                    x_xi    = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
                    y_xi    = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

                    alpha   = x_eta ** 2 + y_eta ** 2
                    beta    = x_xi * x_eta + y_xi * y_eta
                    gamma   = x_xi ** 2 + y_xi ** 2

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

                i           = m-1
                x_eta       = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta       = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi        = (X[1, j] - X[i-1, j]) / 2 / d_xi
                y_xi        = (Y[1, j] - Y[i-1, j]) / 2 / d_xi
                alpha       = x_eta ** 2 + y_eta ** 2
                beta        = x_xi * x_eta + y_xi * y_eta
                gamma       = x_xi ** 2 + y_xi ** 2
                Xn[i, j]    = (d_xi * d_eta) ** 2\
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
        Genera malla resolviendo ecuación de Poisson
        metodo = J (Jacobi), GS (Gauss-Seidel), SOR (Sobre-relajacion)
        '''

        # se genera malla antes por algún método algebráico
        self.gen_TFI()

        # se inician variables
        Xn  = self.X
        Yn  = self.Y
        m   = self.M
        n   = self.N

        d_eta   = self.d_eta
        d_xi    = self.d_xi
        omega   = np.longdouble(1.3)  # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. Solución tiende a diverger
            omega = 1     ---> método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. acelera la convergencia.
        '''

        # parámetros de ecuación de Poisson
        Q           = 0
        P           = 0
        I           = 0
        a           = np.longdouble(0.0)
        c           = np.longdouble(0.0)
        aa          = np.longdouble(26.5)
        cc          = np.longdouble(6.5)
        linea_eta   = 0.0
        linea_xi    = 0.5

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
                    x_eta   = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                    y_eta   = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                    x_xi    = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
                    y_xi    = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

                    alpha   = x_eta ** 2 + y_eta ** 2
                    beta    = x_xi * x_eta + y_xi * y_eta
                    gamma   = x_xi ** 2 + y_xi ** 2

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
                    I           = x_xi * y_eta - x_eta * y_xi

                    Xn[i, j]    = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                        * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                            + gamma / (d_eta ** 2) * (X[i, j+1] + X[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                                    - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])
                            + I ** 2 * (P * x_xi + Q * x_eta))
                    Yn[i, j]    = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                        * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                            + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                                    - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])
                            + I**2 * (P * y_xi + Q * y_eta))

                i       = m-1
                x_eta   = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta   = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi    = (X[1, j] - X[i-1, j]) / 2 / d_xi
                y_xi    = (Y[1, j] - Y[i-1, j]) / 2 / d_xi

                alpha   = x_eta ** 2 + y_eta ** 2
                beta    = x_xi * x_eta + y_xi * y_eta
                gamma   = x_xi ** 2 + y_xi ** 2

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

                Xn[i, j]    = (d_xi * d_eta) ** 2\
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
        m       = self.M
        n       = self.N
        X       = self.X
        Y       = self.Y
        d_xi    = self.d_xi
        d_eta   = self.d_eta
        d_s1    = 0.01
        S       = np.zeros((m - 2, m - 2), dtype=object)
        L       = np.zeros((m - 2, m - 2), dtype=object)
        U       = np.zeros((m - 2, m - 2), dtype=object)
        R       = np.zeros((m - 2, 1), dtype=object)
        Z       = np.zeros((m - 2, 1), dtype=object)
        DD      = np.zeros((m - 2, 1), dtype=object)
        Fprev   = 0.05
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
                    L[i, i]     = S[i, i] - S[i, i - 1] @ U[i - 1, i]
                    U[i, i]     = np.identity(2)
                    U[i, i + 1] = np.linalg.inv(L[i, i]) @ S[i, i + 1]
            # se obtienen los valores del vector Z
            i = 0
            Z[0, 0] = np.linalg.inv(L[0, 0]) @ DD[0, 0]
            for i in range(1, m - 2):
                Z[i, 0] = np.linalg.inv(L[i, i])\
                        @ (DD[i, 0] - L[i, i - 1] @ Z[i - 1, 0])

            # se obtienen los valores del vector R
            i       = m - 3
            R[i, 0] = Z[i, 0]
            for i in range(m - 4, -1, -1):
                R[i, 0] = Z[i, 0] - U[i, i + 1] @ Z[i + 1, 0]

            # se asignan las coordenadas X y Y
            for i in range(1, m - 1):
                X[i, j] = R[i - 1, 0][0]
                Y[i, j] = R[i - 1, 0][1]
            x_xi        = (X[1, j - 1] - X[-2, j - 1]) / 2 / d_xi
            y_xi        = (Y[1, j - 1] - Y[-2, j - 1]) / 2 / d_xi
            X[0, j]     = X[0, j - 1] - d_eta / 2 / d_xi * F\
                / (x_xi ** 2 + y_xi ** 2)
            X[0, j]     *= (Y[1, j - 1] - Y[-2, j - 1])
            X[-1, j]    = X[0, j]
            Fprev       = F
        return

    def gen_parabolic(self):
        '''
        Genera malla resolviendo un sistema de ecuaciones parabólicas
        Basado en el reporte técnico de Siladic
        '''

        m = self.M
        n = self.N
        X = self.X
        Y = self.Y

        weight      = 1.9
        delta_limit = self.R - X[0, 0]
        x_line      = np.zeros(n, dtype='float64')
        h           = delta_limit * (1 - weight) / (1 - weight ** ((n - 1)))
        x_line[-1]  = self.R
        x_line[0]   = X[0, 0]
        dd          = x_line[0]

        for i in range(0, n - 1):
            x_line[i]   = dd
            dd          += h * weight ** i

        X[0, :]     = x_line
        X[-1, :]    = x_line

        delta_limit = 1
        G_line      = np.zeros(n, dtype='float64')
        h           = delta_limit * (1 - weight) / (1 - weight ** (n - 1))
        G_line[-1]  = 1
        G_line[0]   = 0
        dd          = G_line[0]

        for i in range(n - 1):
            G_line[i]   = dd
            dd          += h * weight ** i

        # variables del método de solución
        R_          = np.zeros((m - 2,), dtype=object)
        Y_          = np.zeros((m - 2,), dtype=object)
        delta_Q     = np.zeros((m - 2,), dtype=object)
        A_          = np.empty((m-2,), dtype=object)
        B_          = np.empty((m-2,), dtype=object)
        C_          = np.empty((m-2,), dtype=object)
        alpha_      = np.empty((m-2,), dtype=object)
        beta_       = np.empty((m-2,), dtype=object)
        XO          = np.empty((m,), dtype=object)
        YO          = np.empty((m,), dtype=object)

        # resolver ecuaciones gobernantes
        # A * x[i - i, j] + B x[i, j] + C x[i + 1, j] = Dx
        # A * y[i - i, j] + B y[i, j] + C y[i + 1, j] = Dy

        # A = 2 * alpha / (F[i - 1] * (F[i] + F[i - 1]))
        # C = 2 * alpha / (F[i] * (F[i] + F[i - 1]))
        # B = -2 * alpha / (F[i] + F[i - 1]) * (1 / F[i] + 1 / F[i - 1])\
        #       -2 * gamma / (G[j] + g[j - 1]) * (x[i, j - 1] / g[j - 1]\
        #       + XO[i, j+1] / G[j])
        # Ds = -beta * (SO[i + 1, j + 1] - SO[i - 1, j + 1] - s[i + 1, j - 1]\
        #           + s[i - 1, j - 1]) / (F[i] + F[i - 1]) / (G[j] + g[j - 1])\
        #           - 2 * gamma / (G[j] + g[j - 1]) * (s[i, j - 1] / g[j - 1]\
        #           SO[i, j + 1] / G[j])

        # alpha = x_eta ** 2 + y_eta ** 2
        # beta = x_xi * x_eta + y_xi * y_eta
        # gamma = x_xi ** 2 + y_xi ** 2

        # x_xi = (x[i + 1, j] - x[i - 1, j]) / (F[i] + F[i - 1])
        # x_eta = (XO[i, j + 1] - x[i, j - 1]) / (g[j - 1] + G[j])

        # F[*] = deltas en direccion xi
        # G y g = deltas en direccion eta
        # XO y YO = valores de x[i, j + 1] y y[i, j + 1] interpolados entre
        #       las fronteras
        for j in range(1, n - 1):
            Gj      = x_line[-1] - x_line[j]
            gj_1    = x_line[j] - x_line[j - 1]
            print("j = " + str(i))
            print("Gj = " + str(Gj))
            print("gj_1 = " + str(gj_1))
            ###############################################################
            #
            #   Se calculan valores XO y YO imponiendo ortogonalidad
            #
            ###############################################################
            dist = (X[0, -1] - X[0, 0]) ** 2 + (Y[0, -1] - Y[0, 0]) ** 2
            dist **= 0.5
            # se calcula pendiente del cuerpo para obtener la recta normal
            if abs(Y[1, 0] - Y[-2, 0]) >= 0.01\
                    and abs(X[1, 0] - X[-2, 0]) >= 0.01:
                pendiente = (Y[1, 0] - Y[-2, 0])\
                    / (X[1, 0] - X[-2, 0])
                pendiente = - 1 / pendiente
                a_      = 1 + 1 / pendiente ** 2
                b_      = - 2 * Y[0, 0] / pendiente ** 2 - 2 * Y[0, 0]
                c_      = (1 + 1 / pendiente ** 2) * Y[0, 0] ** 2 - dist ** 2
                y_pos   = (-b_ + (b_ ** 2 - 4 * a_ * c_) ** 0.5) / 2 / a_
                y_neg   = (-b_ - (b_ ** 2 - 4 * a_ * c_) ** 0.5) / 2 / a_
                b_recta = Y[0, 0] - pendiente * X[0, 0]
                x_pos   = (y_pos - b_recta) / pendiente
                x_neg   = (y_neg - b_recta) / pendiente
                x_neg   = (y_neg - b_recta) / pendiente
                XO[0]   = x_pos
                YO[0]   = y_pos
            elif abs(Y[1, 0] - Y[-2, 0]) < 0.01:
                XO[0] = X[0, 0]
                YO[0] = Y[0, 0] + dist

            elif abs(X[1, 0] - X[-2, 0]) < 0.01:
                YO[0] = Y[0, 0]
                XO[0] = X[0, 0] + dist

            XO[-1] = XO[0]
            YO[-1] = YO[0]

            for i in range(1, m - 1):
                # se calcula radio desde [i, 0] hasta [i, -1]
                dist = (X[i, -1] - X[i, 0]) ** 2 + (Y[i, -1] - Y[i, 0]) ** 2
                dist **= 0.5
                # se calcula pendiente del cuerpo para obtener la recta normal
                # si no son aprox 0 se calcula pendiente, si no se dan los
                # valores directo, según sea el caso
                if abs(Y[i + 1, 0] - Y[i - 1, 0]) >= 0.01\
                        and abs(X[i + 1, 0] - X[i - 1, 0]) >= 0.01:
                    pendiente = (Y[i + 1, 0] - Y[i - 1, 0])\
                        / (X[i + 1, 0] - X[i - 1, 0])
                    pendiente = - 1 / pendiente
                    a_ = 1 + 1 / pendiente ** 2
                    b_ = - 2 * Y[i, 0] / pendiente ** 2 - 2 * Y[i, 0]
                    c_ = (1 + 1 / pendiente ** 2) * Y[i, 0] ** 2 - dist ** 2
                    y_pos = (-b_ + (b_ ** 2 - 4 * a_ * c_) ** 0.5) / 2 / a_
                    y_neg = (-b_ - (b_ ** 2 - 4 * a_ * c_) ** 0.5) / 2 / a_
                    b_recta = Y[i, 0] - pendiente * X[i, 0]
                    x_pos   = (y_pos - b_recta) / pendiente
                    x_neg   = (y_neg - b_recta) / pendiente
                    if i <= m // 2:
                        YO[i] = y_neg
                        XO[i] = x_neg
                    else:
                        YO[i] = y_pos
                        XO[i] = x_pos

                elif abs(Y[i + 1, 0] - Y[i - 1, 0]) < 0.01:
                    XO[i] = X[i, 0]
                    if i <= m // 2:
                        YO[i] = Y[i, 0] - dist
                    else:
                        YO[i] = Y[i, 0] + dist

                elif abs(X[i + 1, 0] - X[i - 1, 0]) < 0.01:
                    YO[i] = Y[i, 0]
                    if i <= m // 2:
                        XO[i] = X[i, 0] - dist
                    else:
                        XO[i] = X[i, 0] + dist
            ###############################################################
            #
            #   Termina calculo de valores XO y YO
            #
            ###############################################################

            for i in range(1, m - 1):
                ###############################################################
                #
                #   Se calculan las funciones F como:
                #       F = sqrt(deltaX ** 2 + deltaY ** 2)
                #   Siladic página 44 del texto
                #
                ###############################################################
                Fi      = ((X[i + 1, j - 1] - X[i, j - 1]) ** 2
                      + (Y[i + 1, j - 1] - Y[i, j - 1]) ** 2) ** 0.5
                Fi_1    = ((X[i, j - 1] - X[i - 1, j - 1]) ** 2
                        + (Y[i, j - 1] - Y[i - 1, j - 1]) ** 2) ** 0.5

                x_xi    = (X[i + 1, j - 1] - X[i - 1, j - 1]) / (Fi + Fi_1)
                y_xi    = (Y[i + 1, j - 1] - Y[i - 1, j - 1]) / (Fi + Fi_1)
                x_eta   = (XO[i] - X[i, j - 1]) / (gj_1 + Gj)
                y_eta   = (YO[i] - Y[i, j - 1]) / (gj_1 + Gj)

                alpha   = x_eta ** 2 + y_eta ** 2
                beta    = -2 * (x_xi * x_eta + y_xi * y_eta)
                gamma   = x_xi ** 2 + y_xi ** 2

                A   = 2 * alpha / Fi_1 / (Fi + Fi_1)
                B   = -2 * alpha / (Fi + Fi_1) * (1 / Fi + 1 / Fi_1)\
                    - 2 * gamma / (Gj + gj_1) * (1 / Gj + 1 / gj_1)
                C   = 2 * alpha / Fi / (Fi + Fi_1)
                Dx  = - beta * (XO[i + 1] - XO[i - 1]
                               - X[i + 1, j - 1] + X[i - 1, j - 1])\
                    / (Fi + Fi_1) / (Gj + gj_1) - 2 * gamma / (Gj + gj_1)\
                    * (X[i, j - 1] / gj_1 + XO[i] / Gj)
                Dy  = - beta * (YO[i + 1] - YO[i - 1]
                               - Y[i + 1, j - 1] + Y[i - 1, j - 1])\
                    / (Fi + Fi_1) / (Gj + gj_1) - 2 * gamma / (Gj + gj_1)\
                    * (Y[i, j - 1] / gj_1 + YO[i] / Gj)
                # se comienzan a crear las submatrics de la solución
                # S_ * delta_Q = R
                #   S = matriz tridiagonal formada por submatrices A, B y C
                #       para  cada nivel
                # S = LU
                #   L = matriz A * alpha_
                #   U = I * beta_
                # R == [Dx, Dy]
                A_[i - 1] = np.array(([[A, 0], [0, A]]))
                B_[i - 1] = np.array(([[B, 0], [0, B]]))
                C_[i - 1] = np.array(([[C, 0], [0, C]]))
                ###############################################################
                #
                #   Los resultados de A_, B_, C_, D_ parecen tener coherencia
                #   Para un perfil simétrico el valor 0 y el m-3 son iguales
                #
                #   En el caso de R, los valores de Dy son simétricos y de
                #   sentido opuesto, los de Dx son simétricos
                ###############################################################
                if i - 1 == 0:
                    alpha_[0]   = B_[0]
                    beta_[0]    = np.linalg.inv(B_[0]) @ C_[0]
                    beta_[0]    = np.matmul(np.linalg.inv(B_[0]), C_[0])
                else:
                    alpha_[i - 1]   = B_[i - 1] - A_[i - 1] @ beta_[i - 2]
                    beta_[i - 1]    = np.linalg.inv(alpha_[i - 1]) @ C_[i - 1]

                R_[i - 1] = np.array([[Dx], [Dy]])
                ###############################################################
                #
                #   Los valores de alpha_ y beta_ parecen tener sentido
                #
                ###############################################################
            ###################################################################
            #
            #   A_ = todos son de forma 2x2
            #   B_ = todos son de forma 2x2
            #   C_ = todos son de forma 2x2
            #   alpha_ = todos son de forma 2x2
            #   beta_ = todos son de forma 2x2
            #
            #   R_ = todos son de forma 2x1
            #
            ###################################################################

            # se resuelve LY_ = R
            #   se obtiene vector Y_
            Y_[0] = np.linalg.inv(alpha_[0]) @ R_[0]

            for i in range(1, m - 2):
                Y_[i] = np.linalg.inv(alpha_[i]) @ (R_[i] - A_[i] @ Y_[i - 1])
            # se resuelve Y_ = U_ * delta_Q
            # se obtienen valores de delta_Q que son el resultado final
            delta_Q[m - 3] = Y_[m - 3]

            for i in range(m - 4, -1, -1):
                delta_Q[i] = Y_[i] - beta_[i] @ delta_Q[i + 1]

            for i in range(0, m - 2):
                X[i + 1, j] = delta_Q[i][0, 0]
                Y[i + 1, j] = delta_Q[i][1, 0]
            print(delta_Q)
        return

    def tensor(self):
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
        X       = self.X
        Y       = self.Y
        M       = self.M
        N       = self.N
        d_xi    = self.d_xi
        d_eta   = self.d_eta

        x_xi    = np.zeros((M, N))
        x_eta   = np.zeros((M, N))
        y_xi    = np.zeros((M, N))
        y_eta   = np.zeros((M, N))

        # cálculo de derivadas parciales
        # nodos internos
        for j in range(1, N-1):
            x_eta[:-1, j] = (X[:-1, j+1] - X[:-1, j-1]) / 2 / d_eta
            y_eta[:-1, j] = (Y[:-1, j+1] - Y[:-1, j-1]) / 2 / d_eta

        x_eta[:-1, 0]   = (X[:-1, 1] - X[:-1, 0]) / d_eta
        x_eta[:-1, -1]  = (X[:-1, -1] - X[:-1, -2]) / d_eta
        x_eta[-1, :]    = x_eta[0, :]
        y_eta[:-1, 0]   = (Y[:-1, 1] - Y[:-1, 0]) / d_eta
        y_eta[:-1, -1]  = (Y[:-1, -1] - Y[:-1, -2]) / d_eta
        y_eta[-1, :]    = y_eta[0, :]

        for i in range(1, M-1):
            x_xi[i, :] = (X[i+1, :] - X[i-1, :]) / 2 / d_xi
            y_xi[i, :] = (Y[i+1, :] - Y[i-1, :]) / 2 / d_xi

        x_xi[0, :]  = (X[1, :] - X[-2, :]) / 2 / d_xi
        y_xi[0, :]  = (Y[1, :] - Y[-2, :]) / 2 / d_xi
        x_xi[-1, :] = x_xi[0, :]
        y_xi[-1, :] = y_xi[0, :]

        # obteniendo los tensores de la métrica
        J       = (x_xi * y_eta) - (x_eta * y_xi)
        g11I    = x_xi ** 2 + y_xi ** 2
        g12I    = x_xi * x_eta + y_xi * y_eta
        g22I    = x_eta ** 2 + y_eta ** 2
        g11     = g22I / J ** 2
        g12     = -g12I / J ** 2
        g22     = g11I / J ** 2

        C1      = g11I
        A       = g22I
        B       = g12I

        return (g11, g22, g12, J, x_xi, x_eta, y_xi, y_eta, A, B, C1)
