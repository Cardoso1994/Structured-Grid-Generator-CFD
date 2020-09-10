#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cardoso
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import airfoil


def main():
    viridis = cm.get_cmap('viridis')
    magma = cm.get_cmap('magma')
    tuto = pd.read_csv(
        '/home/cardoso/Tesis/su2/jul_20/cp_comp/flow_tutorial.csv')
    malla_c = pd.read_csv(
        '/home/cardoso/Tesis/su2/jul_20/cp_comp/malla_c_alfa_10.csv')
    malla_o = pd.read_csv(
        '/home/cardoso/Tesis/su2/jul_20/cp_comp/malla_o_alfa_10.csv')
    perfil = airfoil.NACA4(0, 0, 12, 1)

    perfil.create_sin(315)
    perfil.x += 0.25
    x_perf = perfil.x
    y_perf = -perfil.y

    fig, ax = plt.subplots(1)
    ax.set_title('Cp vs x/c - alpha = 10°')
    ax.plot(x_perf, 9 * y_perf, 'k')
    ax.plot(tuto["Points:0"], tuto["Pressure_Coefficient"],
             color=viridis(0.0),
             linewidth=1.9, label='NASA')
    ax.plot(0.25 + malla_c["Points:0"], malla_c["Pressure_Coefficient"],
             color=viridis(0.65),
             linewidth=1.7, label='malla C')
    ax.plot(0.25 + malla_o["Points:0"][1:-1],
             malla_o["Pressure_Coefficient"][1:-1], color=magma(0.65),
             linewidth=1.7, label='malla O')
    ax.legend(loc='upper right')
    ax.set_xlabel('x / c')
    ax.set_ylabel('Cp')
    ax.invert_yaxis()
    plt.savefig('/home/cardoso/garbage/cp_comparison_10.png',
                bbox_inches='tight', pad_inches=0.05)
    plt.show()


if __name__ == '__main__':
    main()
