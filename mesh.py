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


# clase para la generación de mallas
class mesh(object):
    
    # variables de clase, controlan el numero de iteraciones máximo
    # así como el error maximo permisible como criterio de convergencia
    it_max = 8000 
    err_max = 1e-6
    
    # método de inicialización de instancias de clase
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        X = matriz cuadrada que contiene todos las coordenadas 'x' de los puntos de la malla
        Y = matriz cuadrada que contiene todos las coordenadas 'y' de los puntos de la malla
        '''
        self.R = np.longdouble(R)
        self.M = M
        self.N = N
        self.archivo = archivo
        
        self.X = np.zeros((M, N ), dtype = np.longdouble)
        self.Y = np.copy(self.X)
        
        
        self.d_xi = np.longdouble(1) # / (self.M - 1))
        self.d_eta = np.longdouble(1) # / (self.N - 1))
        self.tipo = None

    # función para graficar la malla
    def plot(self):
        plt.axis('equal')
        plt.plot(self.X, self.Y, 'k', linewidth = 1.6)
        for i in range(self.M):
            plt.plot(self.X[i, :], self.Y[i, :], 'navy', linewidth = 1.6)
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
        omega = 0.3 # en caso de metodo SOR
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
        a = 0
        c = 0
        aa = 500 # aa en pdf
        cc = 25 ## cc en pdf
        it  = 0
        linea_xi = 0
        linea_eta = 0.5
        
        
        # inicio del método iterativo
        while it < mesh.it_max:
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
                        if np.abs(i / (m-1) - linea_eta) == 0:
                            P = 0
                        else:
                            P = -a * (i / (m-1) - linea_eta) / np.abs(i / (m-1) - linea_eta) * np.exp(-c * np.abs(i / (m-1) - linea_eta))
                        
                        if np.abs(j / (n-1) - linea_xi) == 0:
                            Q = 0
                        else:
                            Q = -aa * (j / (n-1) - linea_xi) / np.abs(j / (n-1) - linea_xi) * np.exp(-cc * np.abs(j / (n-1) - linea_xi))
                        #P = 0
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
                if self.tipo == 'O':
                    alpha = 0.25 * ((X[i, j+1] - X[i, j-1]) ** 2 + (Y[i, j+1] - Y[i, j-1]) ** 2)
                    beta = 0.25 * ( (X[1, j] - X[i-1, j]) * (X[i, j+1] - X[i, j-1]) )\
                           + 0.25 * ( (Y[1, j] - Y[i-1, j]) * (Y[i, j+1] - Y[i, j-1]) )
                    gamma = 0.25 * (X[1, j] - X[i-1, j]) ** 2 + 0.25 * (Y[1, j] - Y[i-1, j]) ** 2
                    if ec == 'P': #and self.tipo == 'O':
                        I = (X[1, j] - X[i-1, j]) * (Y[i, j+1] - Y[i, j-1]) / 4 / d_xi / d_eta + (Y[1, j] - Y[i-1, j]) * (X[i, j+1] - X[i, j-1]) / 4 / d_xi / d_eta
                    #else:
                        #pass
    
                    #if self.tipo == 'O':
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

            if abs(Xn - Xo).max() < mesh.err_max and abs(Yn - Yo).max() < mesh.err_max:
                print(metodo + ': saliendo...')
                print('it=',it)
                break

        self.X = Xn
        self.Y = Yn
        return



