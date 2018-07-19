#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:53:21 2018

@author: cardoso

Define clase mesh. Se generan diferentes subclases para la generación de diferentes tipos de malla.
Se definen procedimientos para los diferentes métodos de generación de las mismas.
"""

import numpy as np
import matplotlib.pyplot as plt
#
#
#

it_max = 8000 #8000
err_max = 1e-6


# clase para la generación de mallas
class mesh(object):
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        X = matriz cuadrada que contiene todos las coordenadas 'x' de los puntos de la malla
        Y = matriz cuadrada que contiene todos las coordenadas 'y' de los puntos de la malla
        '''
        self.R = R
        self.M = M
        self.N = N
        self.archivo = archivo
        self.X = np.zeros((M, N ))
        self.Y = np.copy(self.X)
        self.d_xi = 1 #/ (self.M - 1)
        self.d_eta = 1 #/ (self.N - 1)
        self.tipo = None

    # función para graficar la malla
    def plot(self):
        #plt.figure('Malla O')
        plt.axis('equal')
        plt.plot(self.X, self.Y, 'k', linewidth = 0.8)
        for i in range(self.M):
            plt.plot(self.X[i, :], self.Y[i, :], 'purple', linewidth = 0.8)
        plt.show()

    # genera malla por interpolación polinomial por Lagrange
    # sec 4.2.1 M Farrashkhalvat Grid generation
    def gen_inter_pol(self, eje = 'eta'):
        Xn = np.copy(self.X)
        Yn = np.copy(self.Y)

        if eje == 'eta':
            n = self.N
            eta = np.linspace(0, 1, n)
            for j in range(1, n-1):
                Xn[:, j] = Xn[:, 0] * (1 - eta[j]) + Xn[:, -1] * eta[j]
                Yn[:, j] = Yn[:, 0] * (1 - eta[j]) + Yn[:, -1] * eta[j]
            self.X = Xn
            self.Y = Yn
            return (Xn, Yn)
        elif eje == 'xi':
            m = self.M
            xi = np.linspace(0, 1, m)
            for i in range(1, m-1):
                Xn[i, :] = Xn[0, :] * (1 - xi[i]) + Xn[-1, :] * xi[i]
                Yn[i, :] = Yn[0, :] * (1 - xi[i]) + Yn[-1, :] * xi[i]

    # genera malla por TFI
    # sec 4.3.2 M Farrashkhalvat Grid generation
    def gen_TFI(self):
        Xn = np.copy(self.X)
        Yn = np.copy(self.Y)

        n = self.N
        eta = np.linspace(0, 1, n)
        m = self.M
        xi = np.linspace(0, 1, m)

        for j in range(1, n-1):
            Xn[0, j] = Xn[0, 0] * (1 - eta[j]) + Xn[0, -1] * eta[j]
            Xn[-1, j] = Xn[-1, 0] * (1 - eta[j]) + Xn[-1, -1] * eta[j]
            Yn[0, j] = Yn[0, 0] * (1 - eta[j]) + Yn[0, -1] * eta[j]
            Yn[-1, j] = Yn[-1, 0] * (1 - eta[j]) + Yn[-1, -1] * eta[j]

        for j in range(1, n-1):
            for i in range(1, m-1):
                Xn[i, j] = (1 - xi[i]) * Xn[0, j]  +  xi[i] * Xn[-1, j]  +  (1 - eta[j]) * Xn[i, 0]  +  eta[j] * Xn[i, -1] \
                           - (1 - xi[i]) * (1 - eta[j]) * Xn[0, 0]  -  (1 - xi[i]) * eta[j] * Xn[0, -1] \
                           - (1- eta[j]) * xi[i] * Xn[-1, 0]  -  xi[i] * eta[j] * Xn[-1, -1]

                Yn[i, j] = (1 - xi[i]) * Yn[0, j]  +  xi[i] * Yn[-1, j]  +  (1 - eta[j]) * Yn[i, 0]  +  eta[j] * Yn[i, -1] \
                           - (1 - xi[i]) * (1 - eta[j]) * Yn[0, 0]  -  (1 - xi[i]) * eta[j] * Yn[0, -1] \
                           - (1- eta[j]) * xi[i] * Yn[-1, 0]  -  xi[i] * eta[j] * Yn[-1, -1]
        
        Yn[0, 1:-1] *= 0.85
        Yn[-1, 1:-1] *= 0.85
        self.X = Xn
        self.Y = Yn
        return


    # genera malla por interpolación de Hermite
    # sec 4.2.2 M Farrashkhalvat Grid generation
    def gen_inter_Hermite(self):
        Xn = np.copy(self.X)
        Yn = np.copy(self.Y)
        #m = self.M
        n = self.N
        eta = np.linspace(0, 1, n)

        derX = (Xn[:, -1] - Xn[:, 0]) / 1
        derY = (Yn[:, -1] - Yn[:, 0]) / 200000000
        derX = np.transpose(derX)
        derY = np.transpose(derY)
        # Interpolación de hermite
        for j in range(1, n-1):
            Xn[:, j] = Xn[:, 0] * (2 * eta[j]**3 - 3 * eta[j]**2 + 1) + Xn[:, -1] * (3 * eta[j]**2 - 2 * eta[j]**3) \
                       + derX * (eta[j] ** 3 - 2 * eta[j]**2 + eta[j]) + derX *(eta[j]**3 - eta[j]**2)
            Yn[:, j] = Yn[:, 0] * (2 * eta[j]**3 - 3 * eta[j]**2 + 1) + Yn[:, -1] * (3 * eta[j]**2 - 2 * eta[j]**3) \
                       + derY * (eta[j] ** 3 - 2 * eta[j]**2 + eta[j]) + derY *(eta[j]**3 - eta[j]**2)


        self.X = Xn
        self.Y = Yn
        return (Xn, Yn)


    # funcion para generar mallas mediante Ecuaciones diferenciales parciales (EDP)
    def gen_EDP(self, ec = 'P', metodo = 'SOR'):
        '''
        ecuacion = P (poisson) o L (Laplace) [sistema de ecuaciones elípticas a resolver]
        metodo = J (Jacobi), GS (Gauss-Seidel), SOR (Sobre-relajacion) [métodos iterativos para la solución del sistema de ecs]
        '''
        
        # se genera malla antes por algún método algebráico
        self.gen_TFI()
        
        # se inician variables
        X = np.copy(self.X)
        Y = np.copy(self.Y)
        Xn = np.copy(self.X)
        Yn = np.copy(self.Y)
        Xo = np.copy(self.X)
        Yo = np.copy(self.Y)
        m = self.M
        n = self.N


        d_eta = self.d_eta
        d_xi = self.d_xi
        omega = 0.1 # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. Se ocupa si se sabe que la solución tiende a diverger
            omega = 1     ---> no hay nada que altere, por lo tanto se vuelve el método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. -acelera la convergencia. Se ocupa si de antemano se sabe que la solución converge.
        '''
        
        # parámetros de ecuación de Poisson
        Q = 0
        P = 0
        I = 0
        Ak = 340 # aa en pdf
        Ck = 11.5 ## cc en pdf
        it  = 0
        linea = 0
        
        
        # inicio del método iterativo
        while it < it_max:
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
                    beta = x_xi * x_eta +  y_xi * y_eta
                    gamma = x_xi ** 2 + y_xi ** 2
                    
                    if ec == 'P':
                        if np.abs(j / (n-1) - linea) == 0:
                            Q = 0
                        else:
                            Q = -Ak * (j / (n-1) - linea) / np.abs(j / (n-1) - linea) * np.exp(-Ck * np.abs(j / (n-1) - linea))
                        P = 0
                        I = x_xi * y_eta - x_eta * y_xi
                    else:
                        Q = 0
                        P = 0
                        I = 0

                    Xn[i, j] = (d_xi * d_eta)**2 / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) * ( alpha / (d_xi**2) * (X[i+1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (X[i+1, j+1] - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                             + I**2 * (P * x_xi + Q * x_eta))
                    Yn[i, j] = (d_xi * d_eta)**2 / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) * ( alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j]) + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1] - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])\
                             + I**2 * (P * y_xi + Q * y_eta))

                i = m-1
                alpha = 0.25 * ((X[i, j+1] - X[i, j-1]) ** 2 + (Y[i, j+1] - Y[i, j-1]) ** 2)
                beta = 0.25 * ( (X[1, j] - X[i-1, j]) * (X[i, j+1] - X[i, j-1]) )\
                       + 0.25 * ( (Y[1, j] - Y[i-1, j]) * (Y[i, j+1] - Y[i, j-1]) )
                gamma = 0.25 * (X[1, j] - X[i-1, j]) ** 2 + 0.25 * (Y[1, j] - Y[i-1, j]) ** 2
                if ec == 'P' and self.tipo == 'O':
                    I = (X[1, j] - X[i-1, j]) * (Y[i, j+1] - Y[i, j-1]) / 4 / d_xi / d_eta + (Y[1, j] - Y[i-1, j]) * (X[i, j+1] - X[i, j-1]) / 4 / d_xi / d_eta
                else:
                    pass

                if self.tipo == 'O':
                    Xn[-1, j] = (alpha / (d_xi**2) * (X[1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                               - beta / (2 * d_xi * d_eta) * (X[1, j+1] - X[1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                               + I**2 / 2 *(P *(X[1, j] - X[i-1,j]) / d_xi) + Q * (X[i, j+1] - X[i, j-1]) / d_eta)\
                               / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))
                    Yn[-1, j] = (alpha / (d_xi**2) * (Y[1, j] + Y[i-1, j]) + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])\
                               - beta / (2 * d_xi * d_eta) * (Y[1, j+1] - Yo[1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])\
                               + I**2 / 2 *(P *(Y[1, j] - Y[i-1,j]) / d_xi) + Q * (Y[i, j+1] - Y[i, j-1]) / d_eta)\
                               / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))

            if self.tipo == 'O':
                Xn[0, :] = Xn[-1, :]
                Yn[0, :] = Yn[-1, :]

            # se aplica sobre-relajacion si el metodo es SOR
            if metodo == 'SOR':
                Xn = omega * Xn + (1 - omega) * Xo
                Yn = omega * Yn + (1 - omega) * Yo

            it += 1

            if abs(Xn - Xo).max() < err_max and abs(Yn - Yo).max() < err_max:
                print(metodo + ': saliendo...')
                print('it=',it)
                break

        self.X = Xn
        self.Y = Yn
        return



#
#
#
# subclase de generación de mallas
class mesh_O(mesh):
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        '''
        mesh.__init__(self, R, M, N, archivo)
        self.tipo = 'O'
        # probando para quitar función "fronteras"
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
        theta2 = np.linspace(np.pi, 2* np.pi, points)
        theta = np.concatenate((theta, theta2[1:]))
        del(theta2)
        x = R * np.cos(theta)
        y = R * np.sin(theta)


        x = np.flip(x, 0)
        y = np.flip(y,0)
        # primera columna FI (perfil), ultima columna FE
        self.X[:, -1] = x
        self.Y[:, -1] = y
        self.X[:, 0] = perfil_x
        self.Y[:, 0] = perfil_y
        return


class mesh_C(mesh):
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        '''
        mesh.__init__(self, R, M, N, archivo)
        self.tipo = 'C'
        # probando para quitar función "fronteras"
        self.fronteras()

    def fronteras(self):
        '''
        Genera la frontera externa de la malla así como la interna
        '''
        R = self.R
        M = self.M
        N = self.N

        # cargar datos del perfil
        perfil = np.loadtxt(self.archivo)
        perfil_x = perfil[:, 0]

        perfil_y = perfil[:, 1]
        points = np.shape(perfil_x)[0]
        points1 = (points + 1) // 2 #(points + 1) // 2
        # frontera externa
        theta = np.linspace(3 * np.pi / 2, np.pi, points1)
        theta2 = np.linspace(np.pi, np.pi / 2, points1)
        theta = np.concatenate((theta, theta2[1:]))
        del(theta2, points1)
        # parte circular de FE
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        # se termina FE
        x_line = np.linspace(R * 1.5, 0, ((M - points) // 2 + 1))
        x = np.concatenate((x_line, x[1:]))
        x_line = np.flip(x_line, 0)
        x = np.concatenate((x, x_line[1:]))
        y_line = np.copy(x_line)
        y_line[:] = -R
        y = np.concatenate((y_line, y[1:]))
        y = np.concatenate((y, -y_line[1:]))

        # frontera interna
        x_line = np.linspace(R * 1.5, perfil_x[0], (M - points ) // 2 + 1)
        perfil_x = np.concatenate((x_line, perfil_x[1:]))
        x_line = np.flip(x_line, 0)
        perfil_x = np.concatenate((perfil_x, x_line[1:]))
        y_line[:] = 0
        perfil_y = np.concatenate((y_line, perfil_y[1:]))
        perfil_y = np.concatenate((perfil_y, y_line[1:]))


        # primera columna FI (perfil), ultima columna FE
        self.X[:, -1] = x
        self.Y[:, -1] = y
        self.X[:, 0] = perfil_x
        self.Y[:, 0] = perfil_y
        pass

class mesh_H(mesh):
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna'''

        mesh.__init__(self, R, M, N, archivo)
        self.tipo = 'H'
        # probando para quitar función "fronteras"
        self.fronteras()

    def fronteras(self):
        '''
        Genera la frontera externa de la malla así como la interna
        '''
        R = self.R
        x = np.linspace(-R, R, 3)
        return
