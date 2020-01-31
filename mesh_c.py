#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 1 13:53:21 2018

@author: cardoso

Define subclase mesh_C.
Diversos métodos de generación para mallas tipo C
"""

import numpy as np
import matplotlib.pyplot as plt

from mesh import mesh
import mesh_su2

class mesh_C(mesh):
    #def __init__(self, R, M, N, airfoil, from_file=False):
    def __init__(self, R, N, airfoil, from_file=False):
        '''
        R = radio de la frontera externa, en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        '''

        # M = np.shape(airfoil.x)[0] * 3 // 2 - 1
        M = np.shape(airfoil.x)[0] * 3
        mesh.__init__(self, R, M, N, airfoil)
        self.tipo = 'C'

        if not from_file:
            self.fronteras(airfoil)

        return

    def fronteras(self, airfoil):
        '''
        Genera la frontera externa de la malla así como la interna
        '''
        R = self.R
        M = self.M
        # N = self.N

        # cargar datos del perfil
        perfil = airfoil
        perfil_x = perfil.x
        perfil_y = perfil.y
        points = np.shape(perfil_x)[0]
        points1 = (points + 1) // 2
        print('points')
        print(points)
        print('points1')
        print(points1)
        print('M')
        print(M)

        # frontera externa
        theta = np.linspace(3 * np.pi / 2, np.pi, points1)
        theta2 = np.linspace(np.pi, np.pi / 2, points1)
        theta = np.concatenate((theta, theta2[1:]))
        del(theta2, points1)
        # parte circular de FE
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        # se termina FE

        # cambiar variable x_line para que no sea distribución lineal
        x_line = np.linspace(R * 2.5, 0, ((M - points) // 2 + 1))
        '''
        ***
        *** Prueba para quitar distribucion lineal de puntos
        ***
        '''
        npoints = (M - points) // 2 + 1
        print('npoints')
        print(npoints)
        weight = 1.05
        delta_limit = 2.5 * R
        x_line = np.zeros(npoints, dtype='float64')
        print(delta_limit * (1 - weight))
        print(1 - weight ** (npoints))
        h = delta_limit * (1 - weight) / (1 - weight ** ((npoints - 1)))
        # x_line[0] = perfil_x[0]
        x_line[-1] = 2.5 * R
        dd = x_line[0]
        for i in range(0, npoints - 1):
            x_line[i] = dd
            dd += h * weight ** i
        x_line = np.flip(x_line, 0)
        '''
            Termina prueba
        '''
        # dx = (x_line[-2] - x_line[-1]) / 3.5
        # x_line[1:-1] -= dx
        x = np.concatenate((x_line, x[1:]))
        x_line = np.flip(x_line, 0)
        x = np.concatenate((x, x_line[1:]))
        y_line = np.copy(x_line)
        y_line[:] = -R
        y = np.concatenate((y_line, y[1:]))
        y = np.concatenate((y, -y_line[1:]))

        # frontera interna
        # cambiar variable x_line para que no sea distribución lineal
        x_line = np.linspace(R * 2.5, perfil_x[0], (M - points) // 2 + 1)
        '''
        ***
        *** Prueba para quitar distribucion lineal de puntos
        ***
        '''
        npoints = (M - points) // 2 + 1
        weight = 1.2
        delta_limit = 2.5 * R - perfil_x[0]
        x_line = np.zeros(npoints, dtype='float64')
        h = delta_limit * (1 - weight) / (1 - weight ** (npoints - 1))
        x_line[0] = perfil_x[0]
        x_line[-1] = 2.5 * R
        dd = x_line[0]
        for i in range(0, npoints - 1):
            x_line[i] = dd
            dd += h * weight ** i
        x_line = np.flip(x_line, 0)
        '''
            Termina prueba
        '''
        # dx = (x_line[-2] - x_line[-1]) * 253 / 254
        # x_line[1:-1] -= dx
        perfil_x = np.concatenate((x_line[:-1], perfil_x[:]))
        x_line = np.flip(x_line, 0)
        perfil_x = np.concatenate((perfil_x, x_line[1:]))
        y_line[:] = 0
        perfil_y = np.concatenate((y_line[:-1], perfil_y[:]))
        perfil_y = np.concatenate((perfil_y, y_line[1:]))

        # primera columna FI (perfil), ultima columna FE
        self.X[:, -1] = x
        self.Y[:, -1] = y
        self.X[:, 0] = perfil_x
        self.Y[:, 0] = perfil_y

        return

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
        omega = np.longdouble(1.5)  # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. Solución tiende a diverger
            omega = 1     ---> método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. acelera la convergencia.
        '''

        it = 0
        # inicio del método iterativo
        print("Laplace:")
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

                    Xn[i, j] = (d_xi * d_eta)**2\
                        / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                        * (alpha / (d_xi**2) * (X[i+1, j] + X[i-1, j])
                            + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                                    - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1]))
                    Yn[i, j] = (d_xi * d_eta)**2\
                        / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                        * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                            + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                                    - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1]))

                # se calculan los puntos en la sección de salida de la malla
                # parte inferior a partir del corte
                # se ocupan diferencias finitas "forward" para derivadas
                # respecto a "XI"
                i = 0
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i+1, j] - X[i, j]) / d_xi
                y_xi = (Y[i+1, j] - Y[i, j]) / d_xi

                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2

                Yn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
                    * (alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])
                        - beta / d_xi / d_eta
                        * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1])
                        + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))

                # se calculan los puntos en la sección de salida de la malla
                # parte superior a partir del corte
                # se ocupan diferencias finitas "backward" para derivadas
                # respecto a "XI"
                i = m-1
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i, j] - X[i-1, j]) / d_xi
                y_xi = (Y[i, j] - Y[i-1, j]) / d_xi

                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2

                Yn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
                    * (alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])
                        - beta / d_xi / d_eta
                        * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1])
                        + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            it += 1

            if abs(Xn - Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Laplace: ' + metodo + ': saliendo...')
                print('it =', it)
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

        Xn = self.X
        Yn = self.Y
        m = self.M
        n = self.N

        d_eta = self.d_eta
        d_xi = self.d_xi
        omega = np.longdouble(0.1)  # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. la solución tiende a diverger
            omega = 1     ---> método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. acelera la convergencia.
                        se sabe que la solución converge.
        '''

        # parámetros de ecuación de Poisson
        Q = 0
        P = 0
        I = 0
        a = np.longdouble(0)
        c = np.longdouble(0)
        aa = np.longdouble(0.4)  #0.4
        cc = np.longdouble(1.35)  #3.3
        linea_xi = 0.0
        linea_eta = 0.0

        it = 0

        #####
        #####
        #mesh.err_max = 1e-3
        #####
        #####

        # inicio del método iterativo
        print("Poisson:")
        while it < mesh.it_max:
            print(it, end="\r")
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
                                * np.exp(-c * np.abs(np.longdouble(i /
                                                         (m-1) - linea_xi)))

                    if np.abs(j / (n-1) - linea_eta) == 0:
                        Q = np.longdouble(0)
                    else:
                        Q = -aa * (np.longdouble(j / (n-1) - linea_eta))\
                                / np.abs(np.longdouble(j / (n-1) - linea_eta))\
                                * np.exp(-cc * np.abs(np.longdouble(j /
                                                        (n-1) - linea_eta)))

                    I = x_xi * y_eta - x_eta * y_xi

                    Xn[i, j] = (d_xi * d_eta)**2\
                        / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) \
                        * (alpha / (d_xi**2) * (X[i+1, j] + X[i-1, j])
                            + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])
                            - beta / (2 * d_xi * d_eta)
                            * (X[i+1, j+1] - X[i+1, j-1]
                                + X[i-1, j-1] - X[i-1, j+1])
                            + I**2 * (P * x_xi + Q * x_eta))
                    Yn[i, j] = (d_xi * d_eta)**2\
                        / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                        * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                            + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                            - beta / (2 * d_xi * d_eta)
                            * (Y[i+1, j+1] - Y[i+1, j-1]
                                + Y[i-1, j-1] - Y[i-1, j+1])
                            + I**2 * (P * y_xi + Q * y_eta))

                # se calculan los puntos en la sección de salida de la malla
                # parte inferior a partir del corte
                # se ocupan diferencias finitas "forward" para derivadas
                # respecto a "XI"
                i = 0
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i+1, j] - X[i, j]) / d_xi
                y_xi = (Y[i+1, j] - Y[i, j]) / d_xi

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
                    Q = np.longdouble(0)
                else:
                    Q = -aa * (np.longdouble(j / (n-1) - linea_eta))\
                            / np.abs(np.longdouble(j / (n-1) - linea_eta))\
                            * np.exp(-cc
                                * np.abs(np.longdouble(j / (n-1) - linea_eta)))
                I = x_xi * y_eta - x_eta * y_xi

                Yn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
                    * (alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])
                        - beta / d_xi / d_eta
                        * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1])
                        + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
                        + I**2 * (P * y_xi + Q * y_eta))

                # se calculan los puntos en la sección de salida de la malla
                # parte superior a partir del corte
                # se ocupan diferencias finitas "backward" para derivadas
                # respecto a "XI"
                i = m-1
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i, j] - X[i-1, j]) / d_xi
                y_xi = (Y[i, j] - Y[i-1, j]) / d_xi

                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2

                if np.abs(i / (m-1) - linea_xi) == 0:
                    P = np.longdouble(0)
                else:
                    P = -a * (np.longdouble(i / (m-1) - linea_xi))\
                            / np.abs(np.longdouble(i / (m-1) - linea_xi))\
                            * np.exp(-c * np.abs(np.longdouble(i / (m-1)
                                                               - linea_xi)))

                if np.abs(j / (n-1) - linea_eta) == 0:
                    Q = np.longdouble(0)
                else:
                    Q = - aa * (np.longdouble(j / (n-1) - linea_eta))\
                            / np.abs(np.longdouble(j / (n-1) - linea_eta))\
                            * np.exp(-cc * np.abs(np.longdouble(j / (n-1)
                                                                - linea_eta)))
                I = x_xi * y_eta - x_eta * y_xi

                Yn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * gamma * d_xi ** 2 - alpha * d_eta**2)\
                    * (alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])
                        - beta / d_xi / d_eta
                        * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1])
                        + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
                        + I**2 * (P * y_xi + Q * y_eta))

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            it += 1

            if abs(Xn - Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Poisson: ' + metodo + ': saliendo...')
                print('it=', it)
                break

        self.X = Xn
        self.Y = Yn
        return


# función para la generación de mallas mediante EDP hiperbólicas
    def gen_hyperbolic(self):
        # se inician las variables características de la malla
        m = self.M
        n = self.N
        X = self.X
        Y = self.Y
        d_xi = self.d_xi
        d_eta = self.d_eta
        d_s1 = 0.02
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
                F = F * d_s1 * (1 + 0.025) ** (j - 1)
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
                    dd -= (CC @ np.array([[X[-1, j]], [Y[-1, j]]]))
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
            i = 0
            Y[0, j] = Y[1, j]
            Y[-1, j] = Y[-2, j]
            Fprev = F
        return

    def to_su2(self, filename):
        '''
        convierte malla a formato SU2
        '''

        if self.airfoil_alone == True:
            mesh_su2.to_su2_mesh_c_airfoil(self, filename)
        else:
            mesh_su2.to_su2_mesh_c_airfoil_n_flap(self, filename)

        return
