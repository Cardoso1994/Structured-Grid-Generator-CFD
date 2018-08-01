#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 1 13:53:21 2018

@author: cardoso

Define subclase mesh_O.
Se definen diversos métodos de generación para este tipo de mallas
"""

from mesh import mesh
import numpy as np

class mesh_O(mesh):
    def __init__(self, R, M, N, archivo):
        '''
        R = radio de la frontera externa, ya está en función de la cuerda del perfil
            se asigna ese valor desde el sript main.py
        archivo = archivo con la nube de puntos de la frontera interna
        '''
        mesh.__init__(self, R, M, N, archivo)
        self.tipo = 'O'
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
    
    # funcion para generar mallas mediante Ecuaciones diferenciales parciales (EDP)
    def gen_elliptic_EDP(self, ec = 'P', metodo = 'SOR'):
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
        omega = 0.8 # en caso de metodo SOR
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
        aa = 185 # aa en pdf
        cc = 7.2 ## cc en pdf
        it  = 0
        linea_eta = 0
        linea_xi = 0
        
        
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
                        if np.abs(i / (m-1) - linea_xi) == 0:
                            P = 0
                        else:
                            P = -a * (i / (m-1) - linea_xi) / np.abs(i / (m-1) - linea_eta) * np.exp(-c * np.abs(i / (m-1) - linea_eta))
                        
                        if np.abs(j / (n-1) - linea_eta) == 0:
                            Q = 0
                        else:
                            Q = -aa * (j / (n-1) - linea_eta) / np.abs(j / (n-1) - linea_xi) * np.exp(-cc * np.abs(j / (n-1) - linea_xi))
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
            
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[1, j] - X[i-1, j]) / 2 / d_xi
                y_xi = (Y[1, j] - Y[i-1, j]) / 2 / d_xi
                
                alpha = x_eta **2 + y_eta ** 2
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi **2 + y_xi ** 2
                
                if ec == 'P':
                    if np.abs(i / (m-1) - linea_xi) == 0:
                        P = 0
                    else:
                        P = -a * (i / (m-1) - linea_xi) / np.abs(i / (m-1) - linea_xi) * np.exp(-c * np.abs(i / (m-1) - linea_xi))
                    
                    if np.abs(j / (n-1) - linea_eta) == 0:
                        Q = 0
                    else:
                        Q = -aa * (j / (n-1) - linea_eta) / np.abs(j / (n-1) - linea_eta) * np.exp(-cc * np.abs(j / (n-1) - linea_eta))
                    I = x_xi * y_eta - x_eta * y_xi
                else:
                    Q = 0
                    P = 0
                    I = 0

                
                '''Xn[-1, j] = (alpha / (d_xi**2) * (X[1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                               - beta / (2 * d_xi * d_eta) * (X[1, j+1] - X[1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                               + I**2 / 2 *(P *(X[1, j] - X[i-1,j]) / d_xi) + Q * (X[i, j+1] - X[i, j-1]) / d_eta)\
                               / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))'''
                Xn[i, j] = (d_xi * d_eta)**2 / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) * ( alpha / (d_xi**2) * (X[1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (X[1, j+1] - X[1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                             + I**2 * (P * x_xi + Q * x_eta))
                '''Yn[-1, j] = (alpha / (d_xi**2) * (Y[1, j] + Y[i-1, j]) + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])\
                               - beta / (2 * d_xi * d_eta) * (Y[1, j+1] - Yo[1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])\
                               + I**2 / 2 *(P *(Y[1, j] - Y[i-1,j]) / d_xi) + Q * (Y[i, j+1] - Y[i, j-1]) / d_eta)\
                               / 2 / (alpha / (d_xi**2) + gamma / (d_eta**2))'''
                                  
            
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