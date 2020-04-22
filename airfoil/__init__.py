#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 17 13:35:19 2018

@author: cardoso

Define clase airfoil. Genera perfiles a partir de una nube de puntos
Subclase NACA4 para creación de perfiles NACA serie 4
"""

import numpy as np
import matplotlib.pyplot as plt

class airfoil(object):
    """
    Clase para generar perfiles alares (o cualquier frontera interna).

    Genera la nube de puntos que describen a un perfil alar. Pensada
    inicialmente para importar archivos de texto con informacion de perfiles
    aerodinamicos.
    ...

    Atributos
    ----------
    c : float64
        cuerda aerodinamica
    number : int
        numero de perfil (de izquierda a derecha). Por default 1, en caso de
        existir más perfiles o flaps, se enumeran.
    x : numpy.array
        coordenadas en el eje X de los puntos del perfil
    y : numpy.array
        coordenadas en el eje X de los puntos del perfil
    alone : boolean
        True si es un perfil solo, False si hay 2 o mas perfiles
    is_boundary : [bool, numpy.array]
        False si es perfil solo. Numpy.array que indica para cada punto a que
        numero de perfil pertenece o si no es frontera.
    union : int
        numero de puntos que unen a los perfiles. Cero si es perfil unico

    Metodos
    -------
    create(filename):
        Almacena en los atributos del perfil la nube de puntos que lo describe
    size():
        Regresa el numero de coordenadas del perfil
    plot():
        Grafica el perfil generado
    rotate(degrees):
        Rota el perfil tomando como eje 0.25c el degrees grados. Positivo en
        sentido horario.
    join(other, dx, dy=0, union=4):
        Une dos perfiles alares.
    to_csv(filename):
        Exporta la nube de puntos del perfil a un archivo de valores separados
        por comas.
    """

    def __init__(self, c, number=1):
        self.c              = c
        self.number         = number
        self.x              = None
        self.y              = None
        self.alone          = True
        # self.is_boundary    = None
        self.is_boundary    = False
        self.union          = 0

    """
    Funciones que proporcionan los atributos del objeto
    """
    def get_chord(self):
        return (self.c)

    def get_number(self):
        return (self.number)

    def get_x(self):
        return (np.copy(self.x))

    def get_y(self):
        return (np.copy(self.y))

    def is_alone(self):
        return (self.alone)

    def is_boundary_(self):
        return (self.is_boundary)

    def get_union(self):
        return (self.union)

    # se crea un perfil a partir de un archivo con la nube de puntos
    def create(self, filename):
        """
        Almacena en self.x y self.y las coordenadas que describen al perfil
        aerodinamico, proporcionadas en el archivo filename.
        ...

        Parametros
        ----------
        filename : string
            nombre del archivo del cual se extraen los datos.

        Return
        ------
        None
        """

        c       = self.c
        perf    = np.loadtxt(filename)
        x       = perf[:, 0]
        y       = perf[:, 1]
        del (perf)

        # origen del sistema de coordenadas coincide con c/4
        x       -= 0.25
        x       *= c
        y       *= c

        # se especifica que todos los puntos del perfil son frontera
        is_boundary         = np.ones((np.size(x))) * self.number

        self.x              = x
        self.y              = y
        self.is_boundary    = is_boundary

        return

    def size(self):
        '''
        regresa el numero de puntos que forman el perfil
        '''
        return np.size(self.x)

    def plot(self):
        """
        Grafica el perfil aerodinamico
        ...

        Parametros
        ----------
        None

        Return
        ------
        None
        """

        plt.figure('perfil')
        plt.axis('equal')
        plt.plot(self.x, self.y, 'b')
        plt.show()

    def rotate(self, degrees):
        """
        Rota el perfil deegrees grados.
        Se considera en sentido horario una rotacion positiva.
        La rotacion se realiza tomando como eje de rotacion 0.25c
        ...

        Parametros
        ----------
        degrees : float64
            Grados por los cuales se rota el perfil alar.

        Return
        ------
        None
        """

        size    = self.size()
        rads    = degrees * np.pi / 180 * -1

        x       = np.cos(rads) * self.x - np.sin(rads) * self.y
        y       = np.sin(rads) * self.x + np.cos(rads) * self.y

        self.x  = x
        self.y  = y
        self.y  -= self.y[0]

        return

    def join(self, other, dx, dy=0, union=4):
        """
        Une 2 perfiles aerodinamicos.
        ...

        Parametros
        ----------
        other : airfoil
            perfil aerodinamico que se va a unir al perfil que invoca al
            metodo
        dx : float64
            distancia de separacion en el eje X entre el borde de salida del
            primer perfil y el borde de ataque del segundo.
        dy : float64
            distancia de separacion en el eje Y entre el borde de salida del
            primer perfil y el borde de ataque del segundo.
        union : int
            numero de puntos mediante los cuales se unen los perfiles

        Return
        ------
        None
        """

        self.alone      = False
        self.union      = union
        union           += 2
        x_airfoil       = self.x
        y_airfoil       = self.y
        x_flap          = other.x
        y_flap          = other.y
        size_airfoil    = self.size()
        size_flap       = other.size()

        #reajustando en Y
        dy_flap         = y_flap[size_flap // 2]
        dy_total        = dy_flap + dy
        y_airfoil       += dy_total

        # reajustando en X
        dx_air      = -x_flap[size_flap // 2] + x_airfoil[0]
        dx_total    = dx_air + dx
        x_flap      += dx_total
        x_join      = np.linspace(x_flap[size_flap // 2], x_airfoil[0],
                             num=union)
        y_join      = np.linspace(y_flap[size_flap // 2], y_airfoil[0],
                             num=union)

        x_total = x_flap[:size_flap // 2 + 1]
        y_total = y_flap[:size_flap // 2 + 1]
        x_total = np.concatenate((x_total, x_join[1:-1]))
        y_total = np.concatenate((y_total, y_join[1:-1]))
        x_total = np.concatenate((x_total, x_airfoil))
        y_total = np.concatenate((y_total, y_airfoil))
        x_total = np.concatenate((x_total, np.flip(x_join)[1:-1]))
        y_total = np.concatenate((y_total, np.flip(y_join)[1:-1]))
        x_total = np.concatenate((x_total, x_flap[size_flap // 2:]))
        y_total = np.concatenate((y_total, y_flap[size_flap // 2:]))

        is_boundary                 = np.zeros((np.size(x_total)))
        begin                       = 0
        end                         = size_flap // 2 + 1
        is_boundary[:end]           += other.number
        begin                       = end
        end                         += union - 2
        begin                       = end
        end                         += size_airfoil
        is_boundary[begin : end]    = self.number
        begin                       = end
        end                         += union - 2
        begin                       = end
        end                         += size_flap // 2 + 1
        is_boundary[begin : end]    = other.number

        self.x                      = x_total
        self.y                      = y_total
        self.is_boundary            = is_boundary

        return

    def to_csv(self, filename):
        """
        Exporta las coordenadas que describen al perfil a un archivo .csv
        ...

        Parametros
        ----------
        filename : string
            nombre del archivo en donde se guardara la informacion

        Return
        ------
        None
        """

        x               = self.x
        y               = self.y

        airfoil         = np.zeros((self.size(), 2))
        airfoil[:, 0]   = self.x
        airfoil[:, 1]   = self.y

        np.savetxt(filename, airfoil, delimiter=',')

        return


class NACA4(airfoil):
    """
    Clase para generar perfiles alares de la serie NACA de 4 digitos.

    Subclase de airfoil
    Genera la nube de puntos que describen a un perfil alar de la seria NACA de
    4 digitos
    ...

    Atributos
    ----------
    def __init__(self, m, p, t, c, number=1):
    m : int
        combadura maxima del perfil.
    p : int
        posicion de la combadura maxima
    t : int
        espesor maximo del perfil
    c : float64
        cuerda aerodinamica
    number : int
        numero de perfil (de izquierda a derecha). Por default 1, en caso de
        existir más perfiles o flaps, se enumeran.
    x : numpy.array
        coordenadas en el eje X de los puntos del perfil
    y : numpy.array
        coordenadas en el eje X de los puntos del perfil
    alone : boolean
        True si es un perfil solo, False si hay 2 o mas perfiles
    is_boundary : [bool, numpy.array]
        False si es perfil solo. Numpy.array que indica para cada punto a que
        numero de perfil pertenece o si no es frontera.
    union : int
        numero de puntos que unen a los perfiles. Cero si es perfil unico

    Metodos
    -------
    create_linear():
        crea perfil con distribucion lineal en el eje X
    create_sin():
        crea perfil con distribucion senoidal en el eje X
    """

    def __init__(self, m, p, t, c, number=1):
        airfoil.__init__(self, c, number=number)
        self.m = m / 100
        self.p = p / 10
        self.t = t / 100

        return

    def create_linear(self, points):
        """
        crea perfil con una distribucion lineal de los puntos en el eje X
        ...

        Parametros
        ----------
        points : int
            numero de puntos mediante los cuales se unen los perfiles

        Return
        ------
        None
        """

        points  = (points + 1) // 2
        m       = self.m
        p       = self.p
        t       = self.t
        c       = self.c

        # distribución de los puntos en x a lo largo de la cuerda
        xc      = np.linspace(0, 1, points)
        yt      = np.zeros((points, ))
        yc      = np.zeros((points, ))
        xu      = np.zeros((points, ))
        xl      = np.zeros((points, ))
        yl      = np.zeros((points, ))
        theta   = np.zeros((points, ))
        dydx    = np.zeros((points, ))

        a0      = 0.2969
        a1      = -0.126
        a2      = -0.3516
        a3      = 0.2843
        a4      = -0.1015
        # a4 = -0.1036

        # calculo de la distribución de espesor
        yt = 5 * t * (a0 * xc ** 0.5 + a1 * xc + a2 * xc ** 2 + a3 * xc ** 3
                      + a4 * xc ** 4)

        # si es perfil simétrico
        if m == 0 and p == 0:
            xc *= c
            yt *= c

            xu = np.copy(xc)
            yu = np.copy(yt)
            xl = np.copy(xc)
            yl = -yt
        else:
            # cálculo línea de combadura media
            for i in range(points):
                if xc[i] <= p:
                    yc[i]   = (m / p**2) * (2 * p * xc[i] - xc[i]**2)
                    dydx[i] = 2 * m / p**2 * (p - xc[i])
                else:
                    yc[i]   = m / (1 - p)**2 * ((1 - 2*p) + 2 * p * xc[i]
                                              - xc[i] ** 2)
                    dydx[i] = 2 * m / (1 - p)**2 * (p - xc[i])

            theta   = np.arctan(dydx)
            xu      = xc - yt * np.sin(theta)
            xl      = xc + yt * np.sin(theta)
            yu      = yc + yt * np.cos(theta)
            yl      = yc - yt * np.cos(theta)

            # escalamiento a la dimension de la cuerda
            xu *= c
            yu *= c
            xl *= c
            yl *= c
            xc *= c
            yc *= c
            yt *= c

        # ajuste para que el origen del sistema de coordenadas coincida con c/4
        xu  -= c / 4
        xl  -= c / 4
        xc  -= c / 4

        # exportar los datos a un archivo txt
        xuf = np.copy(xu)
        xuf = np.flip(xuf, 0)
        yuf = np.copy(yu)
        yuf = np.flip(yuf, 0)
        xlf = np.copy(xl[1:])
        ylf = np.copy(yl[1:])
        xp  = np.concatenate((xuf, xlf))
        yp  = np.concatenate((yuf, ylf))

        # se invierten para que comience el perfil por el intrados
        # pasando al extrados  SENTIDO HORARIO
        xp  = np.flip(xp, 0)
        yp  = np.flip(yp, 0)

        xp[-1] = xp[0]
        yp[-1] = yp[0]

        # se especifica que todos los puntos del perfil son frontera
        is_boundary         = np.ones((np.size(xp))) * self.number

        self.x              = xp
        self.y              = yp
        self.is_boundary    = is_boundary

        return

    def create_sin(self, points):
        """
        crea perfil con una distribucion senoidal de los puntos en el eje X

        Proporciona una mayor densidad de puntos en los bordes de ataque y
        salida
        ...

        Parametros
        ----------
        points : int
            numero de puntos mediante los cuales se unen los perfiles

        Return
        ------
        None
        """

        points  = (points + 1) // 2
        m       = self.m
        p       = self.p
        t       = self.t
        c       = self.c

        # distribución de los puntos en x a lo largo de la cuerda
        beta    = np.linspace(0, np.pi, points)
        xc      = (1 - np.cos(beta)) / 2
        yt      = np.zeros((points, ))
        yc      = np.zeros((points, ))
        xu      = np.zeros((points, ))
        xl      = np.zeros((points, ))
        yl      = np.zeros((points, ))
        theta   = np.zeros((points, ))
        dydx    = np.zeros((points, ))

        a0      = 0.2969
        a1      = -0.126
        a2      = -0.3516
        a3      = 0.2843
        a4      = -0.1015
        a4 = -0.1036

        # calculo de la distribución de espesor
        yt = 5 * t * (a0 * xc ** 0.5 + a1 * xc + a2 * xc ** 2 + a3 * xc ** 3
                      + a4 * xc ** 4)

        # si es perfil simétrico
        if m == 0 and p == 0:
            xc *= c
            yt *= c

            xu = np.copy(xc)
            yu = np.copy(yt)
            xl = np.copy(xc)
            yl = -yt
        else:
            # cálculo línea de combadura media
            for i in range(points):
                if xc[i] <= p:
                    yc[i]   = (m / p**2) * (2 * p * xc[i] - xc[i]**2)
                    dydx[i] = 2 * m / p**2 * (p - xc[i])
                else:
                    yc[i]   = m / (1 - p)**2 * ((1 - 2*p) + 2 * p * xc[i]
                                              - xc[i] ** 2)
                    dydx[i] = 2 * m / (1 - p)**2 * (p - xc[i])

            theta   = np.arctan(dydx)
            xu      = xc - yt * np.sin(theta)
            xl      = xc + yt * np.sin(theta)
            yu      = yc + yt * np.cos(theta)
            yl      = yc - yt * np.cos(theta)

            # escalamiento a la dimension de la cuerda
            xu *= c
            yu *= c
            xl *= c
            yl *= c
            xc *= c
            yc *= c
            yt *= c

        # ajuste para que el origen del sistema de coordenadas coincida con c/4
        xu  -= c / 4
        xl  -= c / 4
        xc  -= c / 4

        # exportar los datos a un archivo txt
        xuf = np.copy(xu)
        xuf = np.flip(xuf, 0)
        yuf = np.copy(yu)
        yuf = np.flip(yuf, 0)
        xlf = np.copy(xl[1:])
        ylf = np.copy(yl[1:])
        xp  = np.concatenate((xuf, xlf))
        yp  = np.concatenate((yuf, ylf))

        # se invierten para que comience el perfil por el intrados
        # pasando al extrados  SENTIDO HORARIO
        xp      = np.flip(xp, 0)
        yp      = np.flip(yp, 0)
        perfil  = np.zeros((np.shape(xp)[0], 2))

        # coincidir puntos de borde de salida, perfil cerrado
        xp[0] = xp[-1]
        yp[0] = yp[-1]

        # haciendo coincidir el borde de salida con y = 0
        yp -= yp[0]

        perfil[:, 0]    = xp
        perfil[:, 1]    = yp

        # se especifica que todos los puntos del perfil son frontera
        is_boundary         = np.ones((np.size(xp))) * self.number

        self.x              = perfil[:, 0]
        self.y              = perfil[:, 1]
        self.is_boundary    = is_boundary

        return


class cilindro(airfoil):
    '''
    clase para generar cilindros
    '''

    def __init__(self, c, number=1):
        '''
        ocupa metodo __init__ de airfoil
        '''
        airfoil.__init__(self, c)

    def create(self, points):
        '''
        genera un cilindro
        '''
        theta   = np.linspace(2 * np.pi, np.pi, points)
        theta2  = np.linspace(np.pi, 0, points)
        theta   = np.concatenate((theta, theta2[1:]))
        del(theta2)

        x       = self.c * np.cos(theta)
        y       = self.c * np.sin(theta)

        is_boundary         = np.ones((np.size(x))) * self.number

        self.x              = x
        self.y              = y
        self.is_boundary    = is_boundary

        return
