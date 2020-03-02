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
    '''
    Super clase para perfiles
    '''

    def __init__(self, c, number=1):
        '''
        c = cuerda [m]
        x e y = coordenadas de la nube de puntos que describe el perfil
        number = numero de perfil si existen varios
        '''
        self.c              = c
        self.number         = number
        self.x              = None
        self.y              = None
        self.alone          = True
        self.is_boundary    = None
        self.union          = 0

    '''
    métodos para acceder a los atributos de la clase
    '''
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
        '''
        se crea un perfil a partir de un archivo con la nube de puntos
        filename = nombre del archivo con los datos del perfil a importar
        '''
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
        '''
        grafica el perfil aerodinámico
        '''
        plt.figure('perfil')
        plt.axis('equal')
        plt.plot(self.x, self.y, 'b')
        plt.show()

    def rotate(self, degrees):
        '''
            rotación del perfil por 'degrees' grados en sentido horario
                (rotación positiva de perfiles aerodinámicos)
        '''
        size    = self.size()
        rads    = degrees * np.pi / 180 * -1

        x       = np.cos(rads) * self.x - np.sin(rads) * self.y
        y       = np.sin(rads) * self.x + np.cos(rads) * self.y

        self.x  = x
        self.y  = y
        self.y  -= self.y[0]

        return

    def join(self, other, dx, dy=0, union=4):
        '''
            une dos perfiles aerodinámicos. Para el análisis de external
                airfoil flaps
            self = perfil aerodinámico [airfoil]
            other = flap [airfoil]
            dx y dy = distancias en x e y respectivamente entre borde de
                salida del perfil y borde de ataque del flap
            union = número de puntos que unen al perfil y al flap
        '''
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
        '''
        Exporta nube de puntos a archivo CSV
        '''
        x               = self.x
        y               = self.y

        airfoil         = np.zeros((self.size(), 2))
        airfoil[:, 0]   = self.x
        airfoil[:, 1]   = self.y

        np.savetxt(filename, airfoil, delimiter=',')

        return


class NACA4(airfoil):
    '''
        subclase de airfoil.
        Genera perfiles de la serie NACA de 4 dígitos
    '''

    def __init__(self, m, p, t, c, number=1):
        '''
            m = combadura máxima, se divide entre 100
            p = posición de la combadura máxima, se divide entre 10
            t = espesor máximo del perfil, en porcentaje de la cuerda
            c = cuerda del perfil [m]
        '''
        airfoil.__init__(self, c, number=number)
        self.m = m / 100
        self.p = p / 10
        self.t = t / 100

        return

    def create_linear(self, points):
        '''
        Crea un perfil NACA4 con una distribución lineal
        points = número de puntos para el perfil
        '''
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
        '''
        Crea un perfil NACA4 con una distribución no lineal mediante
            una función senoidal.
        Mayor densidad de puntos en bordes de ataque y de salida

        points = número de puntos para el perfil
        '''
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
