#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: cardoso

@mail:   marcoacardosom@gmail.com

Archivo main. Ejemplo de ejecucion de codigo.
"""

# importar dependencias de sistema
import numpy as np
import matplotlib.pyplot as plt

# importar modulos del programa
import airfoil
import mesh
import mesh_c
import mesh_o
import mesh_su2
from potential import potential_flow_o, velocity, pressure,\
    lift_n_drag, streamlines, potential_flow_o_n
import util


########################
# PERFIL AERODINaMICO
########################
# importar desde archivo
airfoil_points = 549
c = 1
filename = 'nombre_archivo_nube_de_puntos.csv'
perfil = airfoil.airfoil(c)
perfil.create(filename)

# perfil NACA 4
m = 0
p = 0
t = 12
c = 1
perfil = airfoil.NACA4(m, p, t, c)
perfil.create_sin(airfoil_points)


#########
# FLAP
#########
# importar desde archivo
filename_flap = 'nombre_archivo_nube_de_puntos_flap.csv'
flap = airfoil.airfoil(filename, number=2)

# flap NACA 4
m_f = 2
p_f = 4
t_f = 12
c_f = 0.2 * c
flap = airfoil.NACA4(m_f, p_f, t_f, c_f, number=2)
flap.create_sin(airfoil_points)
flap.rotate(15)

# union de perfil y flap
union = 39
dx = 0.055
dy = 0.05
perfil.join(flap, dx, dy, union=union)

# rotacion del angulo de ataque del perfil.
# Configuracion completa
alpha = 5
perfil.rotate(alpha)


##########
# MALLA
##########
# tipo de malla (C, O)
malla_tipo = 'O'

# dimensiones de la malla. N (\eta), airfoil_points = M (\xi)
N = 275

# dimension de frontera externa
R = 400 * c

if malla_tipo == 'O':
    mallaNACA = mesh_o.mesh_O(R, N, perfil)
elif malla_tipo == 'C':
    mallaNACA = mesh_c.mesh_C(R, N, perfil)

# metodos de generacion de malla
metodo_iterativo = 'SOR' # 'GS', 'J'
mallaNACA.gen_Laplace_n(metodo='SOR')
# mallaNACA.gen_Laplace_v(metodo='SOR')
# mallaNACA.gen_Laplace(metodo='SOR')
mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.15, aa=1500, cc=12,
                        linea_eta=0)
# mallaNACA.gen_Poisson_v(metodo='SOR', omega=0.15, aa=1500, cc=12,
    # linea_eta=0)
# mallaNACA.gen_Poisson(metodo='SOR', omega=0.15, aa=1500, cc=12,
    # linea_eta=0)

# importar malla
malla_importada = 'nombre_archivo_malla.txt_mesh'
mallaNACA = util.from_txt_mesh(malla_importada)

# grafica de malla
mallaNACA.plot()


########################
# CALIDAD DE LA MALLA
########################
aspect_ratio = mallaNACA.get_aspect_ratio()
skew = mallaNACA.get_skew()
cmap = 'viridis'

# grafica aspect_ratio
plt.figure('aspect_ratio')
plt.title('aspect_ratio')
plt.pcolormesh(mallaNACA.X, mallaNACA.Y, aspect_ratio, cmap=cmap)

# guardar grafica
plt.savefig('nombre_de_imagen.png',
            bbox_inches='tight', pad_inches=0.05)

# mostrar grafica
plt.show()

# grafica skew
plt.figure('skew')
plt.title('skew')
plt.pcolormesh(mallaNACA.X, mallaNACA.Y, skew, cmap=cmap)

# guardar grafica
plt.savefig('nombre_de_imagen.png',
            bbox_inches='tight', pad_inches=0.05)

# mostrar grafica
plt.show()


########
# SU2
########
malla_su2 = 'nombre_archivo_malla.su2'
mallaNACA.to_su2(malla_su2)


##############################################
# FLUJO POTENCIAL. Unicamente mallas tipo O
##############################################
# variables del flujo
t_inf = 293.15 # [K]
p_inf = 101325  # [Pa]
v_inf = 48 # [m / s]
alpha = 5
gamma = 1.4
cp_ = 1007
mu = 18.25e-6
Rg = cp_ * (gamma - 1) / gamma
d_inf = p_inf / (Rg * t_inf)
h_inf = cp_ * t_inf
c_inf = (gamma * p_inf / d_inf) ** 0.5

# relaciones isentropicas
h0 = h_inf + 0.5 * v_inf ** 2
d0 = d_inf * (1 + (gamma - 1) / 2
              * (v_inf / c_inf) ** 2) ** (1 / (gamma - 1))
p0 = p_inf * (d0 / d_inf) ** gamma
mach_inf = v_inf / c_inf
Re = v_inf * c * d_inf / mu

mach_inf = v_inf / c_inf
Re = v_inf * c * d_inf / mu
alphas = [-4, -2, 0, 2, 4, 6, 8, 10]
if mach_inf > 0.8:
    print('Las condiciones de flujo son invalidas')
    exit()

    for alpha_ in alphas:
        alpha = int(alpha_)
        (phi, C, theta, IMA) = potential_flow_o_n(d0, h0, gamma,
                                                  mach_inf, v_inf,
                                                 alpha, mallaNACA)
        (u, v) = velocity(alpha, C, mach_inf, theta, mallaNACA,
                          phi, v_inf)
        (cp, p) = pressure(u, v, v_inf, d_inf, gamma, p_inf, p0,
                           d0, h0)
        (psi, mach) = streamlines(u, v, gamma, h0, d0, p, mallaNACA)
        (L, _) = lift_n_drag(mallaNACA, cp, alpha, c)

        plt.figure('potential')
        plt.contour(mallaNACA.X, mallaNACA.Y, phi, 95, cmap='viridis')
        plt.colorbar()
        plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
        plt.plot(mallaNACA.X[:, -1], mallaNACA.Y[:, -1], 'k')
        plt.axis('equal')

        plt.figure('pressure')
        plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
        plt.contourf(mallaNACA.X, mallaNACA.Y, cp, 75, cmap='viridis')
        plt.colorbar()
        plt.axis('equal')

        plt.figure('streamlines')
        plt.plot(mallaNACA.X[:, 0], mallaNACA.Y[:, 0], 'k')
        plt.contour(mallaNACA.X, mallaNACA.Y, np.real(psi), 195,
                    cmap='viridis')
        plt.colorbar()
        plt.axis('equal')

        plt.draw()
        plt.show()
