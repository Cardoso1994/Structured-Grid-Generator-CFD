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

it_max = 8000
err_max = 1e-5


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
        self.d_xi = 1 / (self.N - 1)
        self.d_eta = 1 / (self.M - 1)
        self.tipo = None
    
    # genera malla por interpolación polinomial por Lagrange
    # sec 4.2.1 M Farrashkhalvat Grid generation
    def gen_inter_pol(self):
        # Xn para nivel de iteración "new".
        Xn = np.copy(self.X)
        Yn = np.copy(self.Y)
        
        #m = self.M
        n = self.N
        
        xi = np.linspace(0, 1, n)
        for j in range(1, n-1):
            Xn[:, j] = Xn[:, 0] * (1 - xi[j]) + Xn[:, -1] * xi[j]
            Yn[:, j] = Yn[:, 0] * (1 - xi[j]) + Yn[:, -1] * xi[j]
        self.X = Xn
        self.Y = Yn
        return
    
    # genera malla por interpolación de Hermite
    # sec 4.2.2 M Farrashkhalvat Grid generation
    def gen_inter_Hermite(self):
        Xn = np.copy(self.X)
        Yn = np.copy(self.Y)
        m = self.M
        n = self.N
        
        xi = np.linspace(0, 1, n)
        
        Xp = np.zeros((np.size(Xn, 0), 2))
        Yp = np.zeros((np.size(Xn, 0), 2))
        
        # se calculan las derivadas mediante diferencias finitas.
        for j in range(1, n-1):
            # Para los primero puntos (y últimos), mediante diferencia "forward"
            '''Xp[0, 0] = (Xn[1, 0] - Xn[0, 0]) / (Yn[1, 0] - Yn[0, 0])
            Xp[0, 1] = (Xn[1, 1] - Xn[0, 1]) / (Yn[1, 1] - Yn[0, 1])
            Yp[0, 0] = (Yn[1, 0] - Yn[0, 0]) / (Xn[1, 0] - Xn[0, 0])
            Yp[0, 1] = (Yn[1, 1] - Yn[0, 1]) / (Xn[1, 1] - Xn[0, 1])
            Xp[-1, 0] = (Xn[1, 0] - Xn[0, 0]) / (Yn[1, 0] - Yn[0, 0])
            Xp[-1, 1] = (Xn[1, 1] - Xn[0, 1]) / (Yn[1, 1] - Yn[0, 1])
            Yp[-1, 0] = (Yn[1, 0] - Yn[0, 0]) / (Xn[1, 0] - Xn[0, 0])
            Yp[-1, 1] = (Yn[1, 1] - Yn[0, 1]) / (Xn[1, 1] - Xn[0, 1])'''
            '''Xp[0, 0] = 0
            Xp[0, 1] = 0
            Yp[0, 0] = 0
            Yp[0, 1] = 0
            Xp[-1, 0] = Xp[0 ,0]
            Xp[-1, 1] = Xp[0, 1]
            Yp[-1, 0] = Yp[0, 0]
            Yp[-1, 1] = Yp[0, 1]'''
            # Ecuacion de anderson para fronteras diferencias finitas
            Xp[0, 1] = (-3 * Xn[0, 1] + 4 * Xn[-1, 1] - Xn[-2, 1]) / (2/3) / (Yn[-2, 1] - Yn[0, 1])
            Xp[0, 0] = (-3 * Xn[0, 0] + 4 * Xn[-1, 0] - Xn[-2, 0]) / (2/3) / (Yn[-2, 0] - Yn[0, 0])
            Yp[0, 1] = (-3 * Yn[0, 1] + 4 * Yn[-1, 1] - Yn[-2, 1]) / (2/3) / (Xn[-2, 1] - Xn[0, 1])
            Yp[0, 0] = (-3 * Yn[0, 0] + 4 * Yn[-1, 0] - Yn[-2, 0]) / (2/3) / (Xn[-2, 0] - Xn[0, 0])
            Xp[-1, 0] = Xp[0 ,0]
            Xp[-1, 1] = Xp[0, 1]
            Yp[-1, 0] = Yp[0, 0]
            Yp[-1, 1] = Yp[0, 1]
            
            
            '''Xp[1, 0] = 0
            Xp[1, 1] = 0
            Yp[1, 0] = 0
            Yp[1, 1] = 0
            Xp[-2, 0] = Xp[1 ,0]
            Xp[-2, 1] = Xp[1, 1]
            Yp[-2, 0] = Yp[1, 0]
            Yp[-2, 1] = -Yp[1, 1]'''
            
            # Para la parte superior de la malla, mediante diferencias centradas
            for i in range(1, m // 2):
                Xp[i, 0] = (Xn[i+1, 0] - Xn[i-1, 0]) / (Yn[i+1, 0] - Yn[i-1, 0])
                Xp[i, 1] = (Xn[i+1, 1] - Xn[i-1, 1]) / (Yn[i+1, 1] - Yn[i-1, 1])
                Yp[i, 0] = (Yn[i+1, 0] - Yn[i-1, 0]) / (Xn[i+1, 0] - Xn[i-1, 0])
                Yp[i, 1] = (Yn[i+1, 1] - Yn[i-1, 1]) / (Xn[i+1, 1] - Xn[i-1, 1])
                
            # Para el borde de ataque (n//2), diferencia "forward"
            i = m // 2
            '''Xp[i, 0] = (Xn[i+1, 0] - Xn[i, 0]) / (Yn[i+1, 0] - Yn[i, 0])
            Xp[i, 1] = (Xn[i+1, 1] - Xn[i, 1]) / (Yn[i+1, 1] - Yn[i, 1])
            Yp[i, 0] = (Yn[i+1, 0] - Yn[i, 0]) / (Xn[i+1, 0] - Xn[i, 0])
            Yp[i, 1] = (Yn[i+1, 1] - Yn[i, 1]) / (Xn[i+1, 1] - Xn[i, 1])'''
            Xp[i, 0] = 0
            Xp[i, 1] = 0
            Yp[i, 0] = 0
            Yp[i, 1] = 0
            
            # Para la parte inferior de la malla, mediante diferencias centradas
            for i in range(m // 2 + 1, m - 1):
                Xp[i, 0] = (Xn[i+1, 0] - Xn[i-1, 0]) / (Yn[i+1, 0] - Yn[i-1, 0])
                Xp[i, 1] = (Xn[i+1, 1] - Xn[i-1, 1]) / (Yn[i+1, 1] - Yn[i-1, 1])
                Yp[i, 0] = (Yn[i+1, 0] - Yn[i-1, 0]) / (Xn[i+1, 0] - Xn[i-1, 0])
                Yp[i, 1] = (Yn[i+1, 1] - Yn[i-1, 1]) / (Xn[i+1, 1] - Xn[i-1, 1])
                
            # Interpolación de hermite
            Xn[:, j] = Xn[:, 0] * (2 * xi[j]**3 - 3 * xi[j]**2 + 1) + Xn[:, -1] * (3 * xi[j]**2 - 2 * xi[j]**3) \
                       + Xp[:, 0] * (xi[j] ** 3 - 2 * xi[j]**2 + xi[j]) + Xp[:, 1] *(xi[j]**3 - xi[j]**2)
            Yn[:, j] = Yn[:, 0] * (2 * xi[j]**3 - 3 * xi[j]**2 + 1) + Yn[:, -1] * (3 * xi[j]**2 - 2 * xi[j]**3) \
                       + Yp[:, 0] * (xi[j] ** 3 - 2 * xi[j]**2 + xi[j]) + Yp[:, 1] *(xi[j]**3 - xi[j]**2)
            
        
        self.X = Xn
        self.Y = Yn
        return (Xn, Yn)
    
    # funcion para generar mallas mediante Ecuaciones diferenciales parciales (EDP)
    def gen_EDP(self, ec = 'P', metodo = 'SOR'):
        '''
        ecuacion = P (poisson) o L (Laplace) [sistema de ecuaciones elípticas a resolver]
        metodo = J (Jacobi), GS (Gauss-Seidel), SOR (Sobre-relajacion) [métodos iterativos para la solución del sistema de ecs]
        '''
        X = np.copy(self.X)
        Y = np.copy(self.Y)
        Xn = np.copy(self.X)
        Yn = np.copy(self.Y)
        Xo = np.copy(self.X)
        Yo = np.copy(self.Y)
        
        m = self.M
        n = self.N
        
        # se genera una malla inicial mediante un método algebráico
        # se calcula un deltax y deltay y se suma a la coordenada anterior de x y y respectivamente
        for i in range(1, m - 1):
            for j in range(1, n-1):
                Xn[:, j] = Xn[:, j-1] + (Xn[:, n-1] - Xn[:, 0]) / (n - 1)
                Yn[:, j] = Yn[:, j-1] + (Yn[:, n-1] - Yn[:, 0]) / (n - 1)
        
        
        d_eta = self.d_eta
        d_xi = self.d_xi
        omega = 1.5 # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. Se ocupa si se sabe que la solución tiende a diverger
            omega = 1     ---> no hay nada que altere, por lo tanto se vuelve el método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. -acelera la convergencia. Se ocupa si de antemano se sabe que la solución converge.
        '''
        Q = 0
        P = 0
        I = 0
        ek = 1
        Ak = 1
        Ck = 10
        it  = 0
        linea = 0.09
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
                    alpha = ((X[i, j+1] - X[i, j-1]) / 2) ** 2 + ((Y[i, j+1] - Y[i, j-1]) / 2) ** 2
                    beta = ((X[i+1, j] - X[i-1, j]) / 2) * ( (X[i, j+1] - X[i, j-1]) / 2) \
                            + ((Y[i+1, j] - Y[i-1, j]) / 2) * ( (Y[i, j+1] - Y[i, j-1]) / 2)
                    gamma = ((X[i+1, j] - X[i-1, j]) / 2) ** 2 + ((Y[i+1, j] - Y[i-1, j]) / 2) ** 2
                    if ec == 'P':
                        if np.abs(j / (n-1) - linea) == 0:
                            Q = 0
                        else:
                            Q = -Ak * (j / (n-1) - linea) / np.abs(j / (n-1) - linea) * np.exp(-Ck * np.abs(j / (n-1) - linea))
                        P = 0
                        I = (X[i+1, j] - X[i-1, j]) * (Y[i, j+1] - Y[i, j-1]) / 4 + (Y[i+1, j] - Y[i-1, j]) * (X[i, j+1] - X[i, j-1]) / 4
                    else:
                        Q = 0
                        P = 0
                        I = 0
                    
                    Xn[i, j] = ( alpha / (d_xi**2) * (X[i+1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (X[i+1, j+1] - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                             + I**2 / 2 *(P *(X[i+1, j] - X[i-1,j]) / d_xi) + Q * (X[i, j+1] - X[i, j-1]) / d_eta)\
                            / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))
                    Yn[i, j] = ( alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j]) + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1] - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])\
                             + I**2 / 2 *(P *(Y[i+1, j] - Y[i-1,j]) / d_xi) + Q * (Y[i, j+1] - Y[i, j-1]) / d_eta)\
                            / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))
                
                i = m-1
                alpha = 0.25 * ((X[i, j+1] - X[i, j-1]) ** 2 + (Y[i, j+1] - Y[i, j-1]) ** 2)
                beta = 0.25 * ( (X[1, j] - X[i-1, j]) * (X[i, j+1] - X[i, j-1]) )\
                       + 0.25 * ( (Y[1, j] - Y[i-1, j]) * (Y[i, j+1] - Y[i, j-1]) )
                gamma = 0.25 * (X[1, j] - X[i-1, j]) ** 2 + 0.25 * (Y[1, j] - Y[i-1, j]) ** 2
                if ec == 'P':
                    I = (X[1, j] - X[i-1, j]) * (Y[i, j+1] - Y[i, j-1]) / 4 + (Y[1, j] - Y[i-1, j]) * (X[i, j+1] - X[i, j-1]) / 4
                else:
                    pass
                
                Xn[-1, j] = (alpha / (d_xi**2) * (X[1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                           - beta / (2 * d_xi * d_eta) * (X[1, j+1] - X[1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                           + I**2 / 2 *(P *(X[1, j] - X[i-1,j]) / d_xi) + Q * (X[i, j+1] - X[i, j-1]) / d_eta)\
                           / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))
                Yn[-1, j] = (alpha / (d_xi**2) * (Y[1, j] + Y[i-1, j]) + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])\
                           - beta / (2 * d_xi * d_eta) * (Y[1, j+1] - Yo[1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])\
                           + I**2 / 2 *(P *(Y[1, j] - Y[i-1,j]) / d_xi) + Q * (Y[i, j+1] - Y[i, j-1]) / d_eta)\
                           / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))
            
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
    
    
    
    # función para graficar la malla
    def plot(self):
        #plt.figure('Malla O')
        plt.axis('equal')
        plt.plot(self.X, self.Y, 'k', linewidth = 0.8)
        for i in range(self.M):
            plt.plot(self.X[i, :], self.Y[i, :], 'purple', linewidth = 0.8)
    

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
        theta = np.linspace(2 * np.pi, np.pi, points)
        theta2 = np.linspace(np.pi, 0, points)
        theta = np.concatenate((theta, theta2[1:]))
        del(theta2)
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        
        
        '''#frontera externa con espaciado uniforme en x, en vez de en theta
        
        # frontera externa
        xu = np.linspace(R, -R, points)
        xd = xu[: -1]
        xd = np.flip(xd, 0)
        yu = (R ** 2 - xu ** 2) ** 0.5
        yd = - (R ** 2 - xd ** 2) ** 0.5
        x = np.concatenate((xu, xd))
        y = np.concatenate((yu, yd))'''
        
        x = np.flip(x, 0)
        y = np.flip(y,0)
        
        # primera columna FE, ultima columna FInterna
        self.X[:, 0] = x
        self.Y[:, 0] = y
        self.X[:, -1] = perfil_x
        self.Y[:, -1] = perfil_y
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
        pass