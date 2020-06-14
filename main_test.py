import numpy as np
import matplotlib.pyplot as plt

#import airfoil
from airfoil import NACA4
from mesh_o import mesh_O
from mesh_c import mesh_C
# import mesh

import mesh_su2
from potential import potential_flow_o
import util
# tipo de malla (C, O)
malla = 'O'

'''
densidad de puntos para la malla
eje "XI"
en el caso de malla tipo O, coincide con el número de puntos del perfil
'''
N = 45
union = 25

# points = 11
airfoil_points = 399 # 499
airfoil_points = 79

if malla == 'C':
    points = airfoil_points // 3  # * 2
elif malla == 'O':
    points = airfoil_points

# datos de perfil NACA
m = 0  # combadura
p = 0  # posicion de la combadura
t = 12  # espesor
c = 1  # cuerda [m]

# radio frontera externa
R = 40 * c

# perfil = airfoil.NACA4(m, p, t, c)
perfil = NACA4(m, p, t, c)
perfil.create_sin(points)
flap = NACA4(m, p, t, 0.2 * c, number=2)
flap.create_sin(points)
flap.rotate(5)
perfil.join(flap, dx=0.055, dy=0.05, union=union)

if malla == 'C':
    mallaNACA = mesh_C(R, N, perfil)
elif malla == 'O':
    mallaNACA = mesh_O(R, N, perfil)

# mallaNACA.gen_Poisson_n(metodo='SOR', omega=0.3, aa=18, cc=27, linea_eta=0)
mallaNACA.gen_hyperbolic()
mallaNACA.plot()

mallaNACA.to_su2('/home/desarrollo/garbage/mesh_o.su2')
mallaNACA.to_txt_mesh('/home/desarrollo/garbage/mesh_o.txt_mesh')

mallaNACA.get_skew()
mallaNACA.get_aspect_ratio()


