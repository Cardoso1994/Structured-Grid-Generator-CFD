#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:33:47 2018

@author: cardoso
"""

import numpy as np
import matplotlib.pyplot as plt

import airfoil
import mesh
import mesh_c
import mesh_o
import mesh_su2
from potential import potential_flow_o, velocity,\
                        pressure, streamlines, lift_n_drag
import util
file_name = '/home/desarrollo/tesis_su2/malla_C_flap/mesh_c_flap.txt_mesh'
mallaNACA = util.from_txt_mesh(filename=file_name)
mallaNACA.plot()
exit()
