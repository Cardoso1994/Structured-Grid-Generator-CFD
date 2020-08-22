"""
@author:    Marco Antonio Cardoso Moreno
@mail:      marcoarcardosom@gmail.com

Extiende subclase mesh_O.

Diversos metodos de generacion para mallas tipo O, apoyandose de la libreria
    numba y de metodos de vectorizado
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from mesh import mesh
import mesh_su2

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

    # asiganicion de variable para metodo
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
        while self.airfoil_boundary[union_start] != 0:
            union_start += 1

    mesh.it_max = 750000
    mesh.err_max = 1e-6
    # inicio del metodo iterativo, separa el metodo para perfil con
    # y sin flap
    print(f"Generando malla tipo O.\nDimensiones M: {self.M}"
          + f" N: {self.N}")
    if self.airfoil_alone:
        print("Perfil")
        print("Poisson numba:")
        for it in range(mesh.it_max):
            if (it % 150e3 == 0):
                self.X = np.copy(Xn)
                self.Y = np.copy(Yn)
                self.plot()
                print()

            # imprime informacion
            print('it = ' + str(it) + ' aa = ' + str(aa) + ' cc = '
                  + str(cc)
                  + ' err_x = ' + '{:.3e}'.format(abs(Xn - Xo).max())
                  + ' err_y = ' + '{:.3e}'.format(abs(Yn - Yo).max())
                  + '\t\t', end="\r")

            Xo = Xn.copy()
            Yo = Yn.copy()
            # si el metodo iterativo es Jacobi
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:   # si el metodo es Gauss-Seidel o SOR
                X = Xn
                Y = Yn

            (Xn, Yn) = _gen_Poisson_n(X, Y, self.M, self.N, P_, Q_)

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            if abs(Xn -Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max and it > 10:
                print('Poisson: ' + metodo + ': saliendo...')
                print('it=', it)
                break
    else:
        print("Perfil con flap")
        print("Poisson numba:")
        for it in range(mesh.it_max):
            if (it % 650e3 == 0):
                self.X = np.copy(Xn)
                self.Y = np.copy(Yn)
                self.plot()
                print()

            # imprime informacion
            print('it = ' + str(it) + ' aa = ' + str(aa) + ' cc = '
                  + str(cc)
                  + ' err_x = ' + '{:.3e}'.format(abs(Xn - Xo).max())
                  + ' err_y = ' + '{:.3e}'.format(abs(Yn - Yo).max())
                  + '\t\t', end="\r")

            Xo = Xn.copy()
            Yo = Yn.copy()
            # si el metodo iterativo es Jacobi
            if metodo == 'J':
                X = Xo
                Y = Yo
            else:   # si el metodo es Gauss-Seidel o SOR
                X = Xn
                Y = Yn

            (Xn, Yn) = _gen_Poisson_n_flap(X, Y, self.M, self.N, P_,
                                           Q_, self.airfoil_boundary,
                                           union_start)

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo


            if abs(Xn -Xo).max() < mesh.err_max\
                    and abs(Yn - Yo).max() < mesh.err_max:
                print('Poisson: ' + metodo + ': saliendo...')
                print('it=', it)
                break

    self.X = Xn
    self.Y = Yn

    return (self.X, self.Y)

@jit
def _gen_Poisson_n_flap(X, Y, M, N,  P_, Q_, airfoil_boundary,
                        union_start):
    """
    Resuelve los for loops anidados correspondientes a la solucion de
    la ecuacion de Poisson para generar la malla.

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
        Cada elemento del array indica el punto a que perfil pertenece
        (numero positivo) o cero si no forma parte de una frontera
    union_start : int
        indice "i" en el que inicia la seccion de union entre los
        perfiles

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

        i       = m-1
        x_eta   = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta   = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi    = (X[1, j] - X[i-1, j]) / 2 / d_xi
        y_xi    = (Y[1, j] - Y[i-1, j]) / 2 / d_xi

        alpha   = x_eta ** 2 + y_eta ** 2
        beta    = x_xi * x_eta + y_xi * y_eta
        gamma   = x_xi ** 2 + y_xi ** 2
        I       = x_xi * y_eta - x_eta * y_xi

        X[i, j]    = (d_xi * d_eta) ** 2\
            / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) \
            * (alpha / (d_xi**2) * (X[1, j] + X[i-1, j]) \
                + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1]) \
                - beta / (2 * d_xi * d_eta) \
                * (X[1, j+1] - X[1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                + I**2 * (P_[i-1] * x_xi + Q_[j-1] * x_eta))

    X[0, 1:-1] = X[m-1, 1:-1]

    # seccion de union entre perfiles
    i = union_start
    while airfoil_boundary[i] == 0:
        x_eta = (X[i, 1] - X[-i - 1, 1]) / 2 / d_eta
        y_eta = (Y[i, 1] - Y[-i - 1, 1]) / 2 / d_eta
        x_xi = (X[i+1, 0] - X[i-1, 0]) / 2 / d_xi
        y_xi = (Y[i+1, 0] - Y[i-1, 0]) / 2 / d_xi

        alpha = x_eta ** 2 + y_eta ** 2
        beta = x_xi * x_eta + y_xi * y_eta
        gamma = x_xi ** 2 + y_xi ** 2
        I = x_xi * y_eta - x_eta * y_xi

        X[i, 0]    = (d_xi * d_eta) ** 2 \
            / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
            * (alpha / (d_xi ** 2) * (X[i+1, 0] + X[i-1, 0])
                + gamma / (d_eta ** 2) * (X[i, 1] + X[-i -1, 1])
                - beta / (2 * d_xi * d_eta) * (X[i+1, 1]
                        - X[-i -2, 1] + X[-i, 1] - X[i-1, 1]))
        Y[i, 0]    = (d_xi * d_eta) ** 2\
            / (2 * (alpha * d_eta ** 2 + gamma * d_xi ** 2))\
            * (alpha / (d_xi ** 2) * (Y[i+1, 0] + Y[i-1, 0])
                + gamma / (d_eta ** 2) * (Y[i, 1] + Y[-i -1, 1])
                - beta / (2 * d_xi * d_eta) * (Y[i+1, 1]
                        - Y[-i -2, 1] + Y[-i, 1] - Y[i-1, 1]))

        X[-i -1, 0] = X[i, 0]
        Y[-i -1, 0] = Y[i, 0]
        i += 1

    return (X, Y)


@jit
def _gen_Poisson_n(X, Y, M, N,  P_, Q_):
    """
    Resuelve los for loops anidados correspondientes a la solucion de
    la ecuacion de Poisson para generar la malla.

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

    for j in range(n-2, 0, -1):
        for i in range(1, m-1):
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

        i       = m-1
        x_eta   = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
        y_eta   = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
        x_xi    = (X[1, j] - X[i-1, j]) / 2 / d_xi
        y_xi    = (Y[1, j] - Y[i-1, j]) / 2 / d_xi

        alpha   = x_eta ** 2 + y_eta ** 2
        beta    = x_xi * x_eta + y_xi * y_eta
        gamma   = x_xi ** 2 + y_xi ** 2
        I       = x_xi * y_eta - x_eta * y_xi

        X[i, j]    = (d_xi * d_eta) ** 2\
            / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) \
            * (alpha / (d_xi**2) * (X[1, j] + X[i-1, j]) \
                + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1]) \
                - beta / (2 * d_xi * d_eta) \
                * (X[1, j+1] - X[1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                + I**2 * (P_[i-1] * x_xi + Q_[j-1] * x_eta))

    X[0, 1:-1] = X[m-1, 1:-1]

    return (X, Y)
