#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    Marco Antonio Cardoso Moreno
@mail:      marcoacardosom@gmail.com

Define subclase mesh_C
"""

import numpy as np
import matplotlib.pyplot as plt

from mesh import mesh
from airfoil import airfoil
import mesh_su2

class mesh_C(mesh):
    """
    Clase para generar mallas tipo C y otros calculos de utilidad sobre
        las mismas
    ...

    Atributos
    ----------
    R : float64
        Radio de la frontera externa en la parte circular. La parte rectangular
            se define en funcion de este parametro
    N : int
        Numero de divisiones en el eje eta.
    airfoil : airfoil
        Objeto de la clase airfoil que define toda la frontera interna
    from_file : boolean
        La malla se crea a partir de una malla almacenada en un archivo con
            extension ".txt_mesh" o se genera en ejecucion.
    weight : float
        Parametro para modificar la distribucion de los puntos (funcion
            exponencial)en la seccion rectangular de la malla. A mayor valor,
            mayor cercania a la zona del perfil alar

    Metodos
    -------
    fronteras(airfoil_x, airfoil_y, weight):
        Genera las fronteras interna y externa de la malla
    gen_Laplace(metodo='SOR', omega=1):
        Genera la malla mediante la solucion de la ecuacion de Laplace
    gen_Poisson(metodo='SOR', omega=1, a=0, c=0, linea_xi=0,
                    aa=0, cc=0, linea_eta=0):
        Genera la malla mediante la solucion de la ecuacion de Poisson
    gen_Poisson_v_(self, metodo='SOR', omega=1, a=0, c=0, linea_xi=0,
                    aa=0, cc=0, linea_eta=0):
        Genera la malla mediante la solucion de la ecuacion de Poisson.
        Utiliza vectorizacion, divide la malla en secciones, tanto en xi como
        en eta.
    gen_Poisson_n(self, metodo='SOR', omega=1, a=0, c=0, linea_xi=0,
                    aa=0, cc=0, linea_eta=0):
        Genera la malla mediante la solucion de la ecuacion de Poisson
        Utiliza la libreria numba para acelerar la ejecucion
    to_su2(filename):
        Convierte la malla a formato de SU2
    """

    def __init__(self, R, N, airfoil, from_file=False, weight=1.055):

        m_ = np.shape(airfoil.x)[0] * 3 // 2
        if not airfoil.alone:
            m_ -= 407
        if m_ % 3 == 1:
            M = m_
        else:
            M = m_ - 1
        mesh.__init__(self, R, M, N, airfoil)

        self.tipo = 'C'

        if not from_file:
            self.fronteras(airfoil.x, airfoil.y, weight)

        return

    # importación de métodos de vectorizado y con librería numba
    from mesh_c_performance import gen_Poisson_v_, gen_Poisson_n

    def fronteras(self, airfoil_x, airfoil_y, weight):
        """
        Genera las fronteras interna y externa de la malla
        ...

        Parametros
        ----------
        airfoil_x : numpy.array
            Coordenadas en el eje X de los puntos que definen al perfil alar
        airfoil_y : numpy.array
            Coordenadas en el eje Y de los puntos que definen al perfil alar
        wieght :     float64
            Valor para funcion exponencial que afecta a la distribucion de
            puntos en la zona rectangular. A mayor "wieght", mayor
            concentracion de puntos en la zona cercana al perfil

        Return
        ------
        None
        """

        R = self.R
        M = self.M
        # N = self.N

        # cargar datos del perfil
        # perfil = airfoil
        perfil_x = airfoil_x
        perfil_y = airfoil_y
        points = np.shape(perfil_x)[0]
        points1 = (points + 1) // 2

        # frontera externa
        a = 1.1 * R
        b = R
        exe = (1 - b ** 2 / a ** 2) ** 0.5
        theta = np.linspace(3 * np.pi / 2, np.pi, points1)
        theta2 = np.linspace(np.pi, np.pi / 2, points1)
        theta = np.concatenate((theta, theta2[1:]))
        r = b / (1 - exe ** 2 * np.cos(theta) ** 2) ** 0.5

        # parte circular de FE
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # se termina FE

        x_line = np.linspace(R * 2.5, 0, ((M - points) // 2 + 1))
        npoints = (M - points) // 2 + 1
        delta_limit = 2.5 * R
        x_line = np.zeros(npoints) * 1.0
        h = delta_limit * (1 - weight) / (1 - weight ** ((npoints - 1)))
        x_line[-1] = 2.5 * R
        dd = x_line[0]
        for i in range(0, npoints - 1):
            x_line[i] = dd
            dd += h * weight ** i
        x_line = np.flip(x_line, 0)
        x = np.concatenate((x_line, x[1:]))
        x_line = np.flip(x_line, 0)
        x = np.concatenate((x, x_line[1:]))
        y_line = np.copy(x_line)
        y_line[:] = -R
        y = np.concatenate((y_line, y[1:]))
        y = np.concatenate((y, -y_line[1:]))

        # frontera interna
        x_line = np.linspace(R * 2.5, perfil_x[0], (M - points) // 2 + 1)
        npoints = (M - points) // 2 + 1
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
        perfil_x = np.concatenate((x_line[:-1], perfil_x[:]))
        x_line = np.flip(x_line, 0)
        perfil_x = np.concatenate((perfil_x, x_line[1:]))
        y_line[:] = 0
        perfil_y = np.concatenate((y_line[:-1], perfil_y[:]))
        perfil_y = np.concatenate((perfil_y[:], y_line[1:]))

        # primera columna FI (perfil), ultima columna FE
        self.X[:, self.N-1] = x
        self.Y[:, self.N-1] = y
        self.X[:, 0] = perfil_x
        self.Y[:, 0] = perfil_y

        return

    def gen_Laplace(self, metodo='SOR', omega=1):
        """
        Resuelve la ecuacion de Laplace para generar la malla.

        Metodo clasico, con for loops anidados.
        ...

        Parametros
        ----------
        metodo : str
            Metodo iterativo de solucion. Jacobi (J), Gauss Seidel (GS) y
            sobrerelajacion (SOR)
        omega : float64
            Valor utilizado para acelerar o suavizar la solucion. Solo se
            utiliza si metodo == 'SOR'
            omega < 1 ---> suaviza la solucion
            omega = 1 ---> metodod Gauss Seidel
            omega > 1 ---> acelera la solucion

        Return
        ------
        None
        """

        # aproximacion inicial
        self.gen_TFI()

        # se inician variables
        Xn = self.X
        Yn = self.Y
        m = self.M
        n = self.N

        d_eta = self.d_eta
        d_xi = self.d_xi

        it = 0
        # inicio del método iterativo
        print("Laplace:")
        while it < mesh.it_max:
            print('it = ' + str(it) + '\t', end='\r')
            Xo = np.copy(Xn)
            Yo = np.copy(Yn)

            # si el método iterativo es Jacobi, GS o SOR
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:
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

                # puntos en la sección de salida de la malla
                # parte inferior a partir del corte
                # se ocupan diferencias finitas "forward" para derivadas
                # respecto a "xi"
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

                # puntos en la sección de salida de la malla
                # parte superior a partir del corte
                # se ocupan diferencias finitas "backward" para derivadas
                # respecto a "xi"
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

            # criterio de convergencia
            if abs(Xn - Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Laplace: ' + metodo + ': saliendo...')
                print('it =', it)
                break

            it += 1

        self.X = Xn
        self.Y = Yn
        return

    def gen_Poisson(self, metodo='SOR', omega=1, a=0, c=0, linea_xi=0,
                    aa=0, cc=0, linea_eta=0):
        """
        Resuelve la ecuacion de Poisson para generar la malla.

        Metodo clasico, con for loops anidados.
        ...

        Parametros
        ----------
        metodo : str
            Metodo iterativo de solucion. Jacobi (J), Gauss Seidel (GS) y
            sobrerelajacion (SOR)
        omega : float64
            Valor utilizado para acelerar o suavizar la solucion. Solo se
            utiliza si metodo == 'SOR'
            omega < 1 ---> suaviza la solucion
            omega = 1 ---> metodod Gauss Seidel
            omega > 1 ---> acelera la solucion
        a, c : float64
            valores ocupados para la funcion de forzado P, en el eje xi
        linea_xi : int
            linea  en el eje xi hacia la cual se realiza el forzado.
            0 <= linea_xi <= self.M
        aa, cc : float64
            valores ocupados para la funcion de forzado Q, en el eje eta
        linea_eta : int
            linea  en el eje eta hacia la cual se realiza el forzado.
            0 <= linea_eta <= self.N

        Return
        ------
        None
        """

        # aproximacion inicial
        self.gen_TFI()

        return
        Xn      = self.X
        Yn      = self.Y
        Xo = np.copy(Xn)
        Yo = np.copy(Yn)
        m       = self.M
        n       = self.N

        d_eta   = self.d_eta
        d_xi    = self.d_xi
        P_ = np.arange(1, m)
        Q_ = np.arange(1, n)
        P_ = -a * (np.longdouble(P_ / (m-1) - linea_xi))\
                                / np.abs(np.longdouble(P_ / (m-1) - linea_xi))\
                                * np.exp(-c * np.abs(np.longdouble(P_ /
                                                         (m-1) - linea_xi)))
        Q_ = -aa * (np.longdouble(Q_ / (n-1) - linea_eta))\
                                / np.abs(np.longdouble(Q_ / (n-1) - linea_eta))\
                                * np.exp(-cc
                                * np.abs(np.longdouble(Q_ / (n-1) - linea_eta)))

        it = 0
        mesh.it_max = 45e3

        # inicio del método iterativo
        print("Poisson:")
        while it < mesh.it_max:
            # if (it % 10000 == 0):
            if (it % 5000 == 0):
                self.X = np.copy(Xn)
                self.Y = np.copy(Yn)
                self.plot()
                save = input("Save current Mesh: [Y/n] ")
                if save == 'Y' or save == 'y':
                    name = input('name of mesh: ')
                    mallaNACA.to_su2(f"/home/desarrollo/garbage/{name}.su2")
                    mallaNACA.to_txt_mesh(
                        f"/home/desarrollo/garbage/{name}.txt_mesh")

            # printing info
            print('it = ' + str(it) + ' aa = ' + str(aa) + ' cc = ' + str(cc)
                  + ' err_x = ' + '{:.3e}'.format(abs(Xn - Xo).max())
                  + ' err_y = ' + '{:.3e}'.format(abs(Yn - Yo).max())
                  + '\t\t', end="\r")
            Xo = np.copy(Xn)
            Yo = np.copy(Yn)
            # si el método iterativo es Jacobi
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:   # si el método es Gauss-Seidel o SOR
                X = Xn
                Y = Yn

            for j in range(n-2, 0, -1):
                for i in range(1, m-1):
                    x_eta = np.longdouble((X[i, j+1] - X[i, j-1]) / 2 / d_eta)
                    y_eta = np.longdouble((Y[i, j+1] - Y[i, j-1]) / 2 / d_eta)
                    x_xi = np.longdouble((X[i+1, j] - X[i-1, j]) / 2 / d_xi)
                    y_xi = np.longdouble((Y[i+1, j] - Y[i-1, j]) / 2 / d_xi)

                    alpha = np.longdouble(x_eta ** 2 + y_eta ** 2)
                    beta = np.longdouble(x_xi * x_eta + y_xi * y_eta)
                    gamma = np.longdouble(x_xi ** 2 + y_xi ** 2)
                    I = x_xi * y_eta - x_eta * y_xi

                    Xn[i, j]    = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                        * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                            + gamma / (d_eta ** 2) * (X[i, j+1] + X[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                                    - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])
                            + I ** 2 * (P_[i-1] * x_xi + Q_[j-1] * x_eta))
                    Yn[i, j]    = (d_xi * d_eta) ** 2\
                        / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                        * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                            + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                            - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                                    - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])
                            + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

                # puntos en la sección de salida de la malla
                # se ocupan diferencias finitas "forward" para derivadas
                # respecto a "xi"
                i = 0
                x_eta = np.longdouble((X[i, j+1] - X[i, j-1]) / 2 / d_eta)
                y_eta = np.longdouble((Y[i, j+1] - Y[i, j-1]) / 2 / d_eta)
                x_xi =  np.longdouble((X[i+1, j] - X[i, j]) / d_xi)
                y_xi =  np.longdouble((Y[i+1, j] - Y[i, j]) / d_xi)

                alpha = np.longdouble(x_eta ** 2 + y_eta ** 2)
                beta =  np.longdouble(x_xi * x_eta + y_xi * y_eta)
                gamma = np.longdouble(x_xi ** 2 + y_xi ** 2)
                I = np.longdouble(x_xi * y_eta - x_eta * y_xi)

                Yn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
                    * (alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])
                        - beta / d_xi / d_eta
                        * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1])
                        + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
                        + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

                # puntos en la sección de salida de la malla
                # se ocupan diferencias finitas "backward" para derivadas
                # respecto a "xi"
                i = m-1
                x_eta = np.longdouble((X[i, j+1] - X[i, j-1]) / 2 / d_eta)
                y_eta = np.longdouble((Y[i, j+1] - Y[i, j-1]) / 2 / d_eta)
                x_xi =  np.longdouble((X[i, j] - X[i-1, j]) / d_xi)
                y_xi =  np.longdouble((Y[i, j] - Y[i-1, j]) / d_xi)

                alpha = np.longdouble(x_eta ** 2 + y_eta ** 2)
                beta =  np.longdouble(x_xi * x_eta + y_xi * y_eta)
                gamma = np.longdouble(x_xi ** 2 + y_xi ** 2)
                I = np.longdouble(x_xi * y_eta - x_eta * y_xi)

                Yn[i, j] = (d_xi * d_eta) ** 2\
                    / (2 * gamma * d_xi ** 2 - alpha * d_eta**2)\
                    * (alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])
                        - beta / d_xi / d_eta
                        * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1])
                        + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
                        + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            # criterio de convergencia
            if abs(Xn - Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Poisson: ' + metodo + ': saliendo...')
                print('it=', it)
                break

            it += 1

        self.X = Xn
        self.Y = Yn
        return


    def gen_hyperbolic(self):
        """
        función para la generación de mallas mediante EDP hiperbólicas

        TODO
        """
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
        """
        Exporta la malla a un archivo de texto en formato de SU2.
        ...

        Parametros
        ----------
        filename : str
            nombre del archivo en el cual se exportara la malla.
            Debe incluir el path (relativo o absoluto)

        Return
        ------
        None
        """

        if self.airfoil_alone == True:
            mesh_su2.to_su2_mesh_c_airfoil(self, filename)
        else:
            mesh_su2.to_su2_mesh_c_airfoil_n_flap(self, filename)

        return
