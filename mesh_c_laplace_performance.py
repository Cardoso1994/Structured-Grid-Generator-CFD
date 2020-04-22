#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    Marco Antonio Cardoso Moreno
@mail:      marcoacardosom@gmail.com

Extiende subclase mesh_C.

Métodos de generación para mallas tipo C mediante la ecuacion de Laplace,
    apoyandose de la libreria numba y de métodos de vectorizado
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from mesh import mesh
import mesh_su2

def gen_Laplace_v_(self, metodo='SOR', omega=1):
    """
    Resuelve la ecuacion de Laplace para generar la malla.

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


    x_eta       = np.zeros((m, n))
    y_eta       = np.zeros((m, n))
    x_xi        = np.zeros((m, n))
    y_xi        = np.zeros((m, n))

    alpha       = np.zeros((m, n))
    beta        = np.zeros((m, n))
    gamma       = np.zeros((m, n))

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

    it = 0
    mesh.it_max = 45e3
    mesh.it_max = 100e3

    # inicio del método iterativo
    print(f"Generando malla tipo C.\nDimensiones M: {self.M} N: {self.N}")
    if self.airfoil_alone:
        print("Perfil")
    else:
        print("Perfil con flap")

    print("Laplace Vectorized:")
    # while it < mesh.it_max:
    while True:
        if (it % 20000 == 0):
            self.X = np.flip(Xn)
            self.Y = np.flip(Yn)
            self.plot()
            print()

        # printing info
        print(f"it = {it} err_x = "
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

                Xn[i, j]      = (d_xi * d_eta) ** 2\
                    / (2 * (alpha[i, j] * d_eta ** 2 \
                            + gamma[i, j] * d_xi ** 2))\
                    * (alpha[i, j] / (d_xi ** 2) * (X[i_p_1, j] \
                                                             + X[i_m_1, j])
                       + gamma[i, j] / (d_eta ** 2) \
                       * (X[i, j_p_1] + X[i, j_m_1]) \
                       - beta[i, j] / (2 * d_xi * d_eta) \
                       * (X[i_p_1, j_p_1] - X[i_p_1, j_m_1] \
                          + X[i_m_1, j_m_1] - X[i_m_1, j_p_1]))
                Yn[i, j]      = (d_xi * d_eta) ** 2\
                    / (2 * (alpha[i, j] * d_eta ** 2 \
                            + gamma[i, j] * d_xi ** 2))\
                    * (alpha[i, j] / (d_xi ** 2) * (Y[i_p_1, j] \
                                                             + Y[i_m_1, j])
                       + gamma[i, j] / (d_eta ** 2) \
                       * (Y[i, j_p_1] + Y[i, j_m_1]) \
                       - beta[i, j] / (2 * d_xi * d_eta) \
                       * (Y[i_p_1, j_p_1] - Y[i_p_1, j_m_1] \
                          + Y[i_m_1, j_m_1] - Y[i_m_1, j_p_1]))

        Yn[0, 1:-1] = Yn[1, 1:-1]
        Yn[-1, 1:-1] = Yn[-2, 1:-1]

        # # se aplica sobre-relajacion si el metodo es SOR
        if metodo == 'SOR':
            Xn = omega * Xn + (1 - omega) * Xo
            Yn = omega * Yn + (1 - omega) * Yo

        # criterio de convergencia
        if abs(Xn - Xo).max() < mesh.err_max\
                and abs(Yn - Yo).max() < mesh.err_max:
            print('Laplace: ' + metodo + ': saliendo...')
            print('it=', it)
            break

        it += 1

    self.X = np.flip(Xn)
    self.Y = np.flip(Yn)
    return


def gen_Laplace_n(self, metodo='SOR', omega=1):
    """
    Resuelve la ecuacion de Laplace para generar la malla.

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
    mesh.it_max = 950e3
    mesh.err_max = 1e-8

    # inicio del metodo iterativo, separar el metodo para perfl con y sin flap
    print(f"Generando malla tipo C. \nDimensiones M: {self.M} N: {self.N}")
    if self.airfoil_alone:
        print("Perfil")
        print("Laplace numba:")

        it = 0
        while it < mesh.it_max:
            if (it % 25000 == 0):
                self.X = np.copy(Xn)
                self.Y = np.copy(Yn)
                self.plot()
                print()

            # imprime informacion
            print('it = ' + str(it)
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

            (Xn, Yn) = _gen_Laplace_n(X, Y, self.M, self.N)

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            it += 1

            if abs(Xn -Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Laplace: ' + metodo + ': saliendo...')
                print('it=', it)
                break
    else:
        print("Perfil con flap")
        print("Laplace numba:")

        it = 0
        while it < mesh.it_max:
            if (it % 25000 == 0):
                self.X = np.copy(Xn)
                self.Y = np.copy(Yn)
                self.plot()
                print()

            # printing info
            print('it = ' + str(it)
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

            (Xn, Yn) = _gen_Laplace_n_flap(X, Y, self.M, self.N,
                                           self.airfoil_boundary, union_start)

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            it += 1

            if abs(Xn -Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Laplace: ' + metodo + ': saliendo...')
                print('it=', it)
                break

    self.X = Xn
    self.Y = Yn
    return (self.X, self.Y)


@jit
def _gen_Laplace_n_flap(X, Y, M, N, airfoil_boundary, union_start):
    """
    Resuelve los for loops anidados correspondientes a la solucion de la
    ecuacion de Laplace para generar la malla.

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

    for j in range(n-2, 0, -1):
        for i in range(1, m-1):
            x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
            y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
            x_xi = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
            y_xi = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

            alpha = x_eta ** 2 + y_eta ** 2
            beta = x_xi * x_eta + y_xi * y_eta
            gamma = x_xi ** 2 + y_xi ** 2

            X[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                    + gamma / (d_eta ** 2) * (X[i, j+1] + X[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                            - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1]))
            Y[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                    + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                            - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1]))

        # se calculan los puntos en la sección de salida de la malla
        i = 0
        x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi =  (X[i+1, j] - X[i, j]) / d_xi
        y_xi =  (Y[i+1, j] - Y[i, j]) / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta =  x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2

        Y[i, j] = (d_xi * d_eta) ** 2\
            / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
            * (alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])
                - beta / d_xi / d_eta
                * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1])
                + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))

        i = m-1
        x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi =  (X[i, j] - X[i-1, j]) / d_xi
        y_xi =  (Y[i, j] - Y[i-1, j]) / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta =  x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2

        Y[i, j] = (d_xi * d_eta) ** 2\
            / (2 * gamma * d_xi ** 2 - alpha * d_eta**2)\
            * (alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])
                - beta / d_xi / d_eta
                * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1])
                + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))

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

        X[i, 0] = (d_xi * d_eta) ** 2 \
            / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2)) \
            * (alpha / (d_xi ** 2) * (X[i+1, 0] + X[i-1, 0])
               + gamma / (d_eta ** 2) * (X[i, 1] + X[-i -1, 1])
               - beta / (2 * d_xi * d_eta) * (X[i+1, 1]
                        - X[-i -2, 1] + X[-i, 1] - X[i-1, 1]))
        Y[i, 0] = (d_xi * d_eta) ** 2 \
            / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2)) \
            * (alpha / (d_xi ** 2) * (Y[i+1, 0] + Y[i-1, 0])
               + gamma / (d_eta ** 2) * (Y[i, 1] + Y[-i -1, 1])
               - beta / (2 * d_xi * d_eta) * (Y[i+1, 1]
                        - Y[-i -2, 1] + Y[-i, 1] - Y[i-1, 1]))

        X[-i -1, 0] = X[i, 0]
        Y[-i -1, 0] = Y[i, 0]
        i += 1
        i_ += 1

    return (X, Y)


@jit
def _gen_Laplace_n(X, Y, M, N):
    """
    Resuelve los for loops anidados correspondientes a la solucion de la
    ecuacion de Laplace para generar la malla.

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

    Return
    ------
    (X, Y) : numpy.array, numpy.array
        Matrices X y Y que describen la malla. Actualizadas.
    """

    d_eta = 1
    d_xi = 1
    m = M
    n = N

    for j in range(n-2, 0, -1):
        for i in range(1, m-1):
            x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
            y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
            x_xi = (X[i+1, j] - X[i-1, j]) / 2 / d_xi
            y_xi = (Y[i+1, j] - Y[i-1, j]) / 2 / d_xi

            alpha = x_eta ** 2 + y_eta ** 2
            beta = x_xi * x_eta + y_xi * y_eta
            gamma = x_xi ** 2 + y_xi ** 2

            X[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
                * (alpha / (d_xi ** 2) * (X[i+1, j] + X[i-1, j])
                    + gamma / (d_eta ** 2) * (X[i, j+1] + X[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (X[i+1, j+1]
                            - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1]))
            Y[i, j]    = (d_xi * d_eta) ** 2\
                / (2 * (alpha * d_eta**2 + gamma * d_xi**2))\
                * (alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j])
                    + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])
                    - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1]
                            - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1]))

        # se calculan los puntos en la sección de salida de la malla
        i = 0
        x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi =  (X[i+1, j] - X[i, j]) / d_xi
        y_xi =  (Y[i+1, j] - Y[i, j]) / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta =  x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2

        Y[i, j] = (d_xi * d_eta) ** 2\
            / (2 * gamma * d_xi ** 2 - alpha * d_eta ** 2)\
            * (alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])
                - beta / d_xi / d_eta
                * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1])
                + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))

        i = m-1
        x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi =  (X[i, j] - X[i-1, j]) / d_xi
        y_xi =  (Y[i, j] - Y[i-1, j]) / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta =  x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2

        Y[i, j] = (d_xi * d_eta) ** 2\
            / (2 * gamma * d_xi ** 2 - alpha * d_eta**2)\
            * (alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])
                - beta / d_xi / d_eta
                * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1])
                + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))

    return (X, Y)
