#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    Marco Antonio Cardoso Moreno
@mail:      marcoacardosom@gmail.com

Extiende subclase mesh_C.

Métodos de generación para mallas tipo C mediante la ecuacion de Poisson,
    apoyandose de la libreria numba y de métodos de vectorizado
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from mesh import mesh
import mesh_su2

def gen_Poisson_v_(self, metodo='SOR', omega=1, a=0, c=0, linea_xi=0,
                aa=0, cc=0, linea_eta=0):
    """
    Resuelve la ecuacion de Poisson para generar la malla.

    Metodo vectorizado. Se divide la malla en secciones, y se resuelve
    una seccion a la vez mediante operaciones vectorizadas a lo largo de
    la seccion.
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
        valores ocupados para la funcion de forzado P, en el eje eta
    linea_eta : int
        linea  en el eje eta hacia la cual se realiza el forzado.
        0 <= linea_eta <= self.N

    Return
    ------
    None
    """

    # aproximacion inicial
    self.gen_TFI()

    # asiganicion de variable para método
    Xn          = np.flip(self.X)
    Yn          = np.flip(self.Y)
    Xo          = np.copy(Xn)
    Yo          = np.copy(Yn)
    m           = self.M
    n           = self.N
    d_eta       = self.d_eta
    d_xi        = self.d_xi

    linea_eta   = n - linea_eta
    linea_xi    = m - linea_xi

    x_eta       = np.zeros((m, n))
    y_eta       = np.zeros((m, n))
    x_xi        = np.zeros((m, n))
    y_xi        = np.zeros((m, n))

    alpha       = np.zeros((m, n))
    beta        = np.zeros((m, n))
    gamma       = np.zeros((m, n))
    I           = np.zeros((m, n))

    # numero de division y asignacion de limites para secciones
    div_eta     = 25
    lim_        = n // div_eta
    lim         = [1]
    for i in range(1, div_eta):
        lim.append(lim_ * i)
    lim.append(n-1)

    div_xi      = 8
    lim_        = m // div_xi
    lim_xi      = [1]
    for i in range(1, div_xi):
        lim_xi.append(lim_ * i)
    lim_xi.append(m-1)


    # cálculo de funciones de forzado
    P_ = np.arange(0, m) * 1.0
    Q_ = np.arange(0, n) * 1.0

    P_[1:-1] = -a * (P_[1:-1] - linea_xi)\
        / np.abs(P_[1:-1] - linea_xi)\
        * np.exp(-c * np.abs(P_[1:-1] - linea_xi))
    Q_[1:-1] = -aa * (Q_[1:-1] - linea_eta)\
        / np.abs(Q_[1:-1] - linea_eta)\
        * np.exp(-cc * np.abs(Q_[1:-1] - linea_eta))

    mask = np.isnan(P_)
    P_[mask] = 0
    mask = np.isnan(Q_)
    Q_[mask] = 0

    mesh.it_max = 450000

    # inicio del método iterativo
    print(f"Generando malla tipo C.\nDimensiones M: {self.M} N: {self.N}")
    if self.airfoil_alone:
        print("Perfil")
    else:
        print("Perfil con flap")

    print("Poisson Vectorized - 3 sections: ")
    # while it < mesh.it_max:
    for it in range(mesh.it_max):
        if (it % 120000 == 0):
            self.X = np.flip(Xn)
            self.Y = np.flip(Yn)
            self.plot()
            print()

        # printing info
        print(f"it = {it} aa = {aa} cc = {cc} err_x = "
              + '{:.3e}'.format(abs(Xn - Xo).max()) + ' err_y = '
              + '{:.3e}'.format(abs(Yn - Yo).max()) + '\t\t', end="\r")

        Xo = np.copy(Xn)
        Yo = np.copy(Yn)

        # si el metodo iterativo es Jacobi ('J') o 'GS' o 'SOR'
        if metodo == 'J':
            X = Xo
            Y = Yo
        else:
            X = Xn
            Y = Yn

        # for loops anidados para recorrer las diferentes secciones
        for j_ in range(div_eta):
            lim_inf = lim[j_]
            lim_sup = lim[j_ + 1]
            j = slice(lim_inf, lim_sup)
            j_p_1 = slice(lim_inf + 1, lim_sup + 1)
            j_m_1 = slice(lim_inf - 1, lim_sup - 1)
            for i_ in range(div_xi):
                lim_inf_xi = lim_xi[i_]
                lim_sup_xi = lim_xi[i_ + 1]
                i = slice(lim_inf_xi, lim_sup_xi)
                i_p_1 = slice(lim_inf_xi + 1, lim_sup_xi + 1)
                i_m_1 = slice(lim_inf_xi - 1, lim_sup_xi - 1)

                x_eta[i, j] = (X[i, j_p_1] \
                                        - X[i, j_m_1]) / 2 / d_eta
                y_eta[i, j] = (Y[i, j_p_1] \
                                        - Y[i, j_m_1]) / 2 / d_eta
                x_xi[i, j]  = (X[i_p_1, j] \
                                        - X[i_m_1, j]) / 2 / d_xi
                y_xi[i, j]  = (Y[i_p_1, j] \
                                    - Y[i_m_1, j]) / 2 / d_xi

                alpha[i, j] = x_eta[i, j]** 2 \
                                             + y_eta[i, j]** 2
                beta[i, j]  = x_xi[i, j] \
                                                     * x_eta[i, j] \
                                                     + y_xi[i, j] \
                                                     * y_eta[i, j]
                gamma[i, j] = x_xi[i, j] ** 2 \
                                                     + y_xi[i, j] ** 2
                I[i, j]     = x_xi[i, j] \
                                                 * y_eta[i, j] \
                                                 - x_eta[i, j] \
                                                 * y_xi[i, j]


                Xn[i, j]      = (d_xi * d_eta) ** 2\
                    / (2 * (alpha[i, j] * d_eta ** 2 \
                            + gamma[i, j] * d_xi ** 2))\
                    * (alpha[i, j] / (d_xi ** 2) * (X[i_p_1, j] \
                                                             + X[i_m_1, j])
                       + gamma[i, j] / (d_eta ** 2) \
                       * (X[i, j_p_1] + X[i, j_m_1]) \
                       - beta[i, j] / (2 * d_xi * d_eta) \
                       * (X[i_p_1, j_p_1] - X[i_p_1, j_m_1] \
                          + X[i_m_1, j_m_1] - X[i_m_1, j_p_1])
                       + I[i, j] ** 2 * (P_[i, None] \
                                                  * x_xi[i, j] \
                                                  + Q_[None, j] \
                                                  * x_eta[i, j]))
                Yn[i, j]      = (d_xi * d_eta) ** 2\
                    / (2 * (alpha[i, j] * d_eta ** 2 \
                            + gamma[i, j] * d_xi ** 2))\
                    * (alpha[i, j] / (d_xi ** 2) * (Y[i_p_1, j] \
                                                             + Y[i_m_1, j])
                       + gamma[i, j] / (d_eta ** 2) \
                       * (Y[i, j_p_1] + Y[i, j_m_1]) \
                       - beta[i, j] / (2 * d_xi * d_eta) \
                       * (Y[i_p_1, j_p_1] - Y[i_p_1, j_m_1] \
                          + Y[i_m_1, j_m_1] - Y[i_m_1, j_p_1])
                       + I[i, j] ** 2 * (P_[i, None] \
                                                  * y_xi[i, j] \
                                                  + Q_[None, j] \
                                                  * y_eta[i, j]))

        Yn[0, 1:-1] = Yn[1, 1:-1]
        Yn[-1, 1:-1] = Yn[-2, 1:-1]

        # # se aplica sobre-relajacion si el metodo es SOR
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

    self.X = np.flip(Xn)
    self.Y = np.flip(Yn)
    return


def gen_Poisson_n(self, metodo='SOR', omega=1, a=0, c=0, linea_xi=0,
                aa=0, cc=0, linea_eta=0):
    """
    Resuelve la ecuacion de Poisson para generar la malla.

    Metodo que se apoya en el uso de la libreria Numba.
    El codigo es el mismo que en el metodo clasico.
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

    Xn = self.X
    Yn = self.Y
    Xo = Xn.copy()
    Yo = Yn.copy()

    m       = self.M
    n       = self.N

    d_eta   = self.d_eta
    d_xi    = self.d_xi

    P_ = np.arange(1, m)
    Q_ = np.arange(1, n)
    P_ = -a * (P_ / (m-1) - linea_xi)\
                            / np.abs(P_ / (m-1) - linea_xi)\
                            * np.exp(-c * np.abs(P_ /
                                                     (m-1) - linea_xi))
    Q_ = -aa * (Q_ / (n-1) - linea_eta)\
                            / np.abs(Q_ / (n-1) - linea_eta)\
                            * np.exp(-cc
                            * np.abs(Q_ / (n-1) - linea_eta))

    mask = np.isnan(P_)
    P_[mask] = 0
    mask = np.isnan(Q_)
    Q_[mask] = 0

    # obteniendo el indice de la union de los perfiles
    if not self.airfoil_alone:
        union_start = 0
        while self.Y[union_start, 0] == 0:
            union_start += 1
        i = 0
        while self.airfoil_boundary[i] != 0:
            union_start += 1
            i += 1
        union_start -= 1

    it = 0
    mesh.it_max = 750000
    mesh.err_max = 1e-6

    # inicio del metodo iterativo, separar el metodo para perfl con y sin flap
    print(f"Generando malla tipo C. \nDimensiones M: {self.M} N: {self.N}")
    if self.airfoil_alone:
        print("Perfil")
        print("Poisson numba:")

        # while it < mesh.it_max:
        for it in range(mesh.it_max):
            if it % 150000 == 0:
                self.X = np.copy(Xn)
                self.Y = np.copy(Yn)
                self.plot()
                print()

            # imprime informacion
            print('it = ' + str(it) + ' aa = ' + str(aa) + ' cc = ' + str(cc)
                  + ' err_x = ' + '{:.3e}'.format(abs(Xn - Xo).max())
                  + ' err_y = ' + '{:.3e}'.format(abs(Yn - Yo).max())
                  + '\t\t', end="\r")

            Xo = Xn.copy()
            Yo = Yn.copy()
            # si el método iterativo es Jacobi
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:   # si el método es Gauss-Seidel o SOR
                X = Xn
                Y = Yn

            (Xn, Yn) = _gen_Poisson_n(X, Y, self.M, self.N, P_, Q_)

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            if abs(Xn -Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Poisson: ' + metodo + ': saliendo...')
                print('it=', it)
                break
    else:
        print("Perfil con flap")
        print("Poisson numba:")

        # while it < mesh.it_max:
        for it in range(mesh.it_max):
            if (it % 150e3 == 0):
                self.X = np.copy(Xn)
                self.Y = np.copy(Yn)
                self.plot()
                print()

            # printing info
            print('it = ' + str(it) + ' aa = ' + str(aa) + ' cc = ' + str(cc)
                  + ' err_x = ' + '{:.3e}'.format(abs(Xn - Xo).max())
                  + ' err_y = ' + '{:.3e}'.format(abs(Yn - Yo).max())
                  + '\t\t', end="\r")

            Xo = Xn.copy()
            Yo = Yn.copy()
            # si el método iterativo es Jacobi
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:   # si el método es Gauss-Seidel o SOR
                X = Xn
                Y = Yn

            (Xn, Yn) = _gen_Poisson_n_flap(X, Y, self.M, self.N, P_, Q_,
                                      self.airfoil_boundary, union_start)

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo


            if abs(Xn -Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('\nPoisson: ' + metodo + ': saliendo...')
                print('it:', it)
                break

    self.X = Xn
    self.Y = Yn
    return (self.X, self.Y)


@jit
def _gen_Poisson_n_flap(X, Y, M, N,  P_, Q_, airfoil_boundary, union_start):
    """
    Resuelve los for loops anidados correspondientes a la solucion de la
    ecuacion de Poisson para generar la malla.

    Metodo que se apoya en el uso de la libreria Numba.
    El codigo es el mismo que en el metodo clasico.
    ...

    Parametros
    ----------
    X : numpy.array
        Matriz que contiene las coordenadas X que describen a la malla
    Y : numpy.array
        Matriz que contiene las coordenadas Y que describen a la malla
    M : int
        Numero de divisiones en el eje xi.
    N : int
        Numero de divisiones en el eje eta.
    P_ : numpy.array
        Valores de la funcion de forzado P para el eje xi
    Q_ : numpy.array
        Valores de la funcion de forzado Q para el eje eta
    airfoil_boundary : numpy.array
        Cada elemento del array indica el punto a que perfil pertenece (numero
        positivo) o cero si no forma parte de una frontera
    union_start : int
        indice "i" en el que inicia la seccion de union entre los perfiles

    Return
    ------
    (X, Y) : numpy.array, numpy.array
        Matrices X y Y que describen la malla. Actualizadas.
    """

    d_eta = 1
    d_xi = 1
    m = M
    n = N

    begin_perfil = 24
    end_perfil = begin_perfil + 99
    limit = 0
    for j in range(n-2, 0, -1):
        Y[0 : begin_perfil - limit, j] = Y[begin_perfil - limit, j]
        Y[end_perfil + limit :, j] = Y[end_perfil + limit - 1, j]
        for i in range(begin_perfil - limit, end_perfil + limit):
        # for i in range(1, m-1):
            x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
            y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
            x_xi = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
            y_xi = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

            alpha = x_eta ** 2 + y_eta ** 2
            beta = x_xi * x_eta + y_xi * y_eta
            gamma = x_xi ** 2 + y_xi ** 2
            I = x_xi * y_eta - x_eta * y_xi

            X[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                    + gamma / (d_eta ** 2) * (X[i, j+1] + X[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                            - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])
                    + I ** 2 * (P_[i-1] * x_xi + Q_[j-1] * x_eta))
            Y[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                    + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                            - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])
                    + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

        # se calculan los puntos en la sección de salida de la malla
        # i = 0
        # x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        # y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        # x_xi =  (X[i+1, j] - X[i, j]) / d_xi
        # y_xi =  (Y[i+1, j] - Y[i, j]) / d_xi

        # alpha = x_eta ** 2 + y_eta ** 2
        # beta =  x_xi * x_eta + y_xi * y_eta
        # gamma = x_xi ** 2 + y_xi ** 2
        # I = x_xi * y_eta - x_eta * y_xi

        # Y[i, j] = (d_xi * d_eta) ** 2\
        #     / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
        #     * (alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])
        #         - beta / d_xi / d_eta
        #         * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1])
        #         + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
        #         + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

        # i = m-1
        # x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        # y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        # x_xi =  (X[i, j] - X[i-1, j]) / d_xi
        # y_xi =  (Y[i, j] - Y[i-1, j]) / d_xi

        # alpha = x_eta ** 2 + y_eta ** 2
        # beta =  x_xi * x_eta + y_xi * y_eta
        # gamma = x_xi ** 2 + y_xi ** 2
        # I = x_xi * y_eta - x_eta * y_xi

        # Y[i, j] = (d_xi * d_eta) ** 2\
        #     / (2 * gamma * d_xi ** 2 - alpha * d_eta**2)\
        #     * (alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])
        #         - beta / d_xi / d_eta
        #         * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1])
        #         + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
        #         + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

    # seccion de union entre perfiles
    i_ = 0
    while airfoil_boundary[i_] != 0:
        i_ += 1
    i = union_start

    while airfoil_boundary[i_] == 0:
        x_eta = (X[i, 1] - X[-i -1, 1]) / 2 / d_eta
        y_eta = (Y[i, 1] - Y[-i -1, 1]) / 2 / d_eta
        x_xi = (X[i+1, 0] - X[i-1, 0]) / 2 / d_xi
        y_xi = (Y[i+1, 0] - Y[i-1, 0]) / 2 / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta = x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2
        I = x_xi * y_eta - x_eta * y_xi

        # X[i, 0] = (d_xi * d_eta) ** 2 \
        #     / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2)) \
        #     * (alpha / (d_xi ** 2) * (X[i+1, 0] + X[i-1, 0])
        #        + gamma / (d_eta ** 2) * (X[i, 1] + X[-i -1, 1])
        #        - beta / (2 * d_xi * d_eta) * (X[i+1, 1]
        #                 - X[-i -2, 1] + X[-i, 1] - X[i-1, 1]))
        Y[i, 0] = (d_xi * d_eta) ** 2 \
            / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2)) \
            * (alpha / (d_xi ** 2) * (Y[i+1, 0] + Y[i-1, 0])
               + gamma / (d_eta ** 2) * (Y[i, 1] + Y[-i -1, 1])
               - beta / (2 * d_xi * d_eta) * (Y[i+1, 1]
                        - Y[-i -2, 1] + Y[-i, 1] - Y[i-1, 1]))

        # X[-i -1, 0] = X[i, 0]
        Y[-i -1, 0] = Y[i, 0]
        i += 1
        i_ += 1

    return (X, Y)


@jit
def _gen_Poisson_n(X, Y, M, N,  P_, Q_):
    """
    Resuelve los for loops anidados correspondientes a la solucion de la
    ecuacion de Poisson para generar la malla.

    Metodo que se apoya en el uso de la libreria Numba.
    El codigo es el mismo que en el metodo clasico.
    ...

    Parametros
    ----------
    X : numpy.array
        Matriz que contiene las coordenadas X que describen a la malla
    Y : numpy.array
        Matriz que contiene las coordenadas Y que describen a la malla
    M : int
        Numero de divisiones en el eje xi.
    N : int
        Numero de divisiones en el eje eta.
    P_ : numpy.array
        Valores de la funcion de forzado P para el eje xi
    Q_ : numpy.array
        Valores de la funcion de forzado Q para el eje eta

    Return
    ------
    (X, Y) : numpy.array, numpy.array
        Matrices X y Y que describen la malla. Actualizadas.
    """

    d_eta = 1
    d_xi = 1
    m = M
    n = N

    begin_perfil = 118
    end_perfil = begin_perfil + 1083
    limit = 0
    for j in range(n-2, 0, -1):
        # for i in range(1, m-1):
        # Y[1:begin_perfil, j] = Y[0, j]
        # Y[end_perfil:-1, j] = Y[-1, j]
        # Y[1 : begin_perfil - 3, j] = Y[0, j]
        # Y[end_perfil + 4 : -1, j] = Y[-1, j]
        Y[0 : begin_perfil - limit, j] = Y[begin_perfil - limit, j]
        Y[end_perfil + limit: , j] = Y[end_perfil + limit - 1, j]
        # for i in range(begin_perfil, end_perfil):
        for i in range(begin_perfil - limit, end_perfil + limit):
            x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
            y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
            x_xi = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
            y_xi = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

            alpha = x_eta ** 2 + y_eta ** 2
            beta = x_xi * x_eta + y_xi * y_eta
            gamma = x_xi ** 2 + y_xi ** 2
            I = x_xi * y_eta - x_eta * y_xi

            X[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                    + gamma / (d_eta ** 2) * (X[i, j+1] + X[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                            - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])
                    + I ** 2 * (P_[i-1] * x_xi + Q_[j-1] * x_eta))
            Y[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                    + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                            - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])
                    + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

        # se calculan los puntos en la sección de salida de la malla
        i = 0
        x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi =  (X[i+1, j] - X[i, j]) / d_xi
        y_xi =  (Y[i+1, j] - Y[i, j]) / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta =  x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2
        I = x_xi * y_eta - x_eta * y_xi

        # Y[i, j] = (d_xi * d_eta) ** 2\
        #     / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
        #     * (alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])
        #         - beta / d_xi / d_eta
        #         * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1])
        #         + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
        #         + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

        i = m-1
        x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi =  (X[i, j] - X[i-1, j]) / d_xi
        y_xi =  (Y[i, j] - Y[i-1, j]) / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta =  x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2
        I = x_xi * y_eta - x_eta * y_xi

        # Y[i, j] = (d_xi * d_eta) ** 2\
        #     / (2 * gamma * d_xi ** 2 - alpha * d_eta**2)\
        #     * (alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])
        #         - beta / d_xi / d_eta
        #         * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1])
        #         + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])
        #         + I**2 * (P_[i-1] * y_xi + Q_[j-1] * y_eta))

    return (X, Y)
