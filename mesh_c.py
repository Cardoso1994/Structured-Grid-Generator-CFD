#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 1 13:53:21 2018

@author: cardoso

Define subclase mesh_C.
Se definen diversos métodos de generación para este tipo de mallas
"""

from mesh import mesh
import numpy as np


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
        #N = self.N

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
        return




    # funcion para generar mallas mediante  ecuación de Laplace. Ecuaciones diferenciales parciales (EDP)
    def gen_Laplace(self, metodo = 'SOR'):
        '''
        metodo = J (Jacobi), GS (Gauss-Seidel), SOR (Sobre-relajacion) [métodos iterativos para la solución del sistema de ecs]
        '''

        # se genera malla antes por algún método algebráico
        self.gen_TFI()

        # se inician variables
        Xn = self.X
        Yn = self.Y
        m = self.M
        n = self.N

        d_eta = self.d_eta
        d_xi = self.d_xi
        omega = np.longdouble(1.4) # en caso de metodo SOR
        '''
        para métodos de relajación:
            0 < omega < 1 ---> bajo-relajación. Se ocupa si se sabe que la solución tiende a diverger
            omega = 1     ---> no hay nada que altere, por lo tanto se vuelve el método Gauss-Seidel
            1 < omega < 2 ---> sobre-relajación. -acelera la convergencia. Se ocupa si de antemano se sabe que la solución converge.
        '''

        it  = 0
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
                    gamma =  x_xi ** 2 + y_xi ** 2

                    Xn[i, j] = (d_xi * d_eta)**2 / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) * ( alpha / (d_xi**2) * (X[i+1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (X[i+1, j+1] - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1]))
                    Yn[i, j] = (d_xi * d_eta)**2 / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) * ( alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j]) + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1] - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1]))

                # se calculan los puntos en la sección de salida de la malla, parte inferior a partir del corte
                # se ocupan diferencias finitas "forward" para derivadas respecto a "XI"
                i = 0
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i+1, j] - X[i, j]) / d_xi
                y_xi = (Y[i+1, j] - Y[i, j]) / d_xi
                
                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta +  y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2
                    
                    
                '''Xn[i, j] = (d_xi * d_eta) ** 2 / (2 * gamma * d_xi ** 2 - alpha * d_eta **2) * ( alpha / d_xi**2 * (X[i+2, j] - 2 * X[i+1, j])\
                          - beta / d_xi / d_eta * (X[i+1, j+1] - X[i+1, j-1] - X[i, j+1] + X[i, j-1]) + gamma / d_eta**2 * (X[i, j+1] + X[i, j-1]) )'''
                Yn[i, j] =(d_xi * d_eta) ** 2 / (2 * gamma * d_xi ** 2 - alpha * d_eta **2) * ( alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])\
                          - beta / d_xi / d_eta * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1]) + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))


                # se calculan los puntos en la sección de salida de la malla, parte superior a partir del corte
                # se ocupan diferencias finitas "backward" para derivadas respecto a "XI"
                i = m-1
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i, j] - X[i-1, j]) / d_xi
                y_xi = (Y[i, j] - Y[i-1, j]) / d_xi

                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta +  y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2

                

                Yn[i, j] = (d_xi * d_eta) ** 2 / (2 * gamma * d_xi ** 2 - alpha * d_eta **2) * ( alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])\
                          - beta / d_xi / d_eta * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1]) + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1]))


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
    



# funcion para generar mallas mediante  ecuación de Poisson. Ecuaciones diferenciales parciales (EDP)
    def gen_Poisson(self, metodo = 'SOR'):
        '''
        metodo = J (Jacobi), GS (Gauss-Seidel), SOR (Sobre-relajacion) [métodos iterativos para la solución del sistema de ecs]
        '''

        # se genera malla antes por algún método algebráico
        self.gen_TFI()
        
        # se inician variables
        Xn = self.X
        Yn = self.Y
        m = self.M
        n = self.N

        d_eta = self.d_eta
        d_xi = self.d_xi
        omega = np.longdouble(0.4) # en caso de metodo SOR
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
        a = np.longdouble(0)
        c = np.longdouble(0)
        aa = np.longdouble(10.7)
        cc = np.longdouble(3.1)
        linea_xi = 0
        linea_eta = 0.5

        it = 0
        
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
                    gamma =  x_xi ** 2 + y_xi ** 2

                    if np.abs( i / (m-1) - linea_eta ) == 0:
                        P = np.longdouble(0)
                    else:
                        P = -a * ( np.longdouble(i / (m-1) - linea_eta) ) / np.abs( np.longdouble(i / (m-1) - linea_eta) ) * np.exp(-c * np.abs( np.longdouble(i / (m-1) - linea_eta)))

                    if np.abs(j / (n-1) - linea_xi) == 0:
                        Q = np.longdouble(0)
                    else:
                        Q = -aa * ( np.longdouble( j / (n-1) - linea_xi) ) / np.abs( np.longdouble(j / (n-1) - linea_xi) ) * np.exp(-cc * np.abs( np.longdouble(j / (n-1) - linea_xi)) )

                    I = x_xi * y_eta - x_eta * y_xi

                    Xn[i, j] = (d_xi * d_eta)**2 / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) * ( alpha / (d_xi**2) * (X[i+1, j] + X[i-1, j]) + gamma / (d_eta**2) * (X[i, j+1] + X[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (X[i+1, j+1] - X[i+1, j-1] + X[i-1, j-1] - X[i-1, j+1])\
                             + I**2 * (P * x_xi + Q * x_eta))
                    Yn[i, j] = (d_xi * d_eta)**2 / (2 * (alpha * d_eta**2 + gamma * d_xi**2)) * ( alpha / (d_xi**2) * (Y[i+1, j] + Y[i-1, j]) + gamma / (d_eta**2) * (Y[i, j+1] + Y[i, j-1])\
                             - beta / (2 * d_xi * d_eta) * (Y[i+1, j+1] - Y[i+1, j-1] + Y[i-1, j-1] - Y[i-1, j+1])\
                             + I**2 * (P * y_xi + Q * y_eta))

                # se calculan los puntos en la sección de salida de la malla, parte inferior a partir del corte
                # se ocupan diferencias finitas "forward" para derivadas respecto a "XI"
                i = 0
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i+1, j] - X[i, j]) / d_xi
                y_xi = (Y[i+1, j] - Y[i, j]) / d_xi
                
                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta +  y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2
                
                if np.abs(i / (m-1) - linea_eta) == 0:
                    P = np.longdouble(0)
                else:
                    P = -a * ( np.longdouble(i / (m-1) - linea_eta) ) / np.abs( np.longdouble(i / (m-1) - linea_eta) ) * np.exp(-c * np.abs( np.longdouble(i / (m-1) - linea_eta) ))

                if np.abs(j / (n-1) - linea_xi) == 0:
                    Q = np.longdouble(0)
                else:
                    Q = -aa * ( np.longdouble(j / (n-1) - linea_xi) ) / np.abs( np.longdouble(j / (n-1) - linea_xi) ) * np.exp(-cc * np.abs( np.longdouble(j / (n-1) - linea_xi) ))
                I = x_xi * y_eta - x_eta * y_xi
                    
                    
                '''Xn[i, j] = (d_xi * d_eta) ** 2 / (2 * gamma * d_xi ** 2 - alpha * d_eta **2) * ( alpha / d_xi**2 * (X[i+2, j] - 2 * X[i+1, j])\
                          - beta / d_xi / d_eta * (X[i+1, j+1] - X[i+1, j-1] - X[i, j+1] + X[i, j-1]) + gamma / d_eta**2 * (X[i, j+1] + X[i, j-1]) )'''
                Yn[i, j] =(d_xi * d_eta) ** 2 / (2 * gamma * d_xi ** 2 - alpha * d_eta **2) * ( alpha / d_xi**2 * (Y[i+2, j] - 2 * Y[i+1, j])\
                          - beta / d_xi / d_eta * (Y[i+1, j+1] - Y[i+1, j-1] - Y[i, j+1] + Y[i, j-1]) + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])\
                          + I**2 * (P * y_xi + Q * y_eta))


                # se calculan los puntos en la sección de salida de la malla, parte superior a partir del corte
                # se ocupan diferencias finitas "backward" para derivadas respecto a "XI"
                i = m-1
                x_eta = (X[i, j+1] - X[i, j-1]) / 2 / d_eta
                y_eta = (Y[i, j+1] - Y[i, j-1]) / 2 / d_eta
                x_xi = (X[i, j] - X[i-1, j]) / d_xi
                y_xi = (Y[i, j] - Y[i-1, j]) / d_xi

                alpha = x_eta ** 2 + y_eta ** 2
                beta = x_xi * x_eta +  y_xi * y_eta
                gamma = x_xi ** 2 + y_xi ** 2

                if np.abs(i / (m-1) - linea_eta) == 0:
                    P = np.longdouble(0)
                else:
                    P = -a * ( np.longdouble(i / (m-1) - linea_eta) ) / np.abs( np.longdouble(i / (m-1) - linea_eta) ) * np.exp(-c * np.abs( np.longdouble(i / (m-1) - linea_eta) ))

                if np.abs(j / (n-1) - linea_xi) == 0:
                    Q = np.longdouble(0)
                else:
                    Q = -aa * ( np.longdouble(j / (n-1) - linea_xi) ) / np.abs( np.longdouble(j / (n-1) - linea_xi) ) * np.exp(-cc * np.abs( np.longdouble(j / (n-1) - linea_xi) ))
                I = x_xi * y_eta - x_eta * y_xi

                Yn[i, j] = (d_xi * d_eta) ** 2 / (2 * gamma * d_xi ** 2 - alpha * d_eta **2) * ( alpha / d_xi**2 * (-2 * Y[i-1, j] + Y[i-2, j])\
                          - beta / d_xi / d_eta * (Y[i, j+1] - Y[i, j-1] - Y[i-1, j+1] + Y[i-1, j-1]) + gamma / d_eta**2 * (Y[i, j+1] + Y[i, j-1])\
                          + I**2 * (P * y_xi + Q * y_eta))


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
