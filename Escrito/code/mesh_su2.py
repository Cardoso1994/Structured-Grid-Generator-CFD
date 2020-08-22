"""
@author:    Marco Antonio Cardoso Moreno
@mail:      marcoacardosom@gmail.com

Scripts para convertir mallas a formato de SU2

Documentacion: https://su2code.github.io/docs/Mesh-File
"""

import numpy as np
import matplotlib.pyplot as plt

from util.helpers import get_size_airfoil, get_size_airfoil_n_flap

def to_su2_mesh_o_airfoil(mesh, filename):
    '''
    Convierte malla de formato propio a formato de SU2
    Para mallas tipo O
    Con solo un perfil (o cualquier geometria)
    '''

    # importa coordenadas de mesh, se quita ultima fila (repetida).
    X           = np.copy(mesh.X)[:-1, :].transpose()
    Y           = np.copy(mesh.Y)[:-1, :].transpose()

    # convirtiendo a arreglo 1D
    X           = X.flatten()
    Y           = Y.flatten()

    NPOIN       = np.shape(X)[0]

    # creando archivo de malla para SU2
    su2_mesh    = open(filename, 'w')

    su2_mesh.write('NDIME= 2\n')
    su2_mesh.write('NPOIN= ' + str(NPOIN) + '\n')

    # se escriben las coordenadas de los nodos
    for i in range(NPOIN):
        su2_mesh.write(str(X[i]) + '\t' + str(Y[i]) + '\n')

    # se escriben las celdas y la conectividad entre nodos que la
    # forman
    NELEM = (mesh.M - 1) * (mesh.N - 1)
    su2_mesh.write('NELEM= ' + str(NELEM) + '\n')
    for i in range(NELEM):
        # condicion para excluir ultimo volumen del nivel. Al terminar
        # vuelta
        if i % (mesh.M - 1) != mesh.M - 2:
            su2_mesh.write('9 ' + str(i) + ' ' + str(i + 1) + ' '
                       + str(i + mesh.M) + ' ' + str(i + mesh.M - 1)
                           + '\n')
        else:
            su2_mesh.write('9 ' + str(i) + ' ' + str(i - (mesh.M - 2))
                           + ' ' + str(i + 1) + ' '
                           + str(i + mesh.M - 1) + '\n')

    # se escriben las fronteras. Primero FE, luego FI
    NMARK = 2
    su2_mesh.write('NMARK= ' + str(NMARK) + '\n')
    su2_mesh.write('MARKER_TAG= farfield\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(mesh.M - 1) + '\n')
    far1 = (mesh.M - 1) * mesh.N
    far0 = far1 - (mesh.M - 1)
    for i in range(far0, far1 - 1):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(far0) + '\n')

    su2_mesh.write('MARKER_TAG= airfoil\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(mesh.M - 1) + '\n')
    for i in range(mesh.M - 2):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(0) + '\n')
    su2_mesh.close()

    return

def to_su2_mesh_o_airfoil_n_flap(mesh, filename):
    '''
    Convierte malla de formato propio a formato de SU2
    Para mallas tipo O
    Para perfiles con external airfoil flap (o en general,
        2 geometrias separadas)
    '''

    size_airfoil, size_flap = get_size_airfoil_n_flap(\
                                    mesh.airfoil_boundary[:-1])

    # creando archivo de malla para SU2
    su2_mesh        = open(filename, 'w')

    M_SU2           = mesh.M - 1
    N_SU2           = mesh.N - 1
    NPOIN           = M_SU2 * N_SU2 + M_SU2 - 2 - mesh.airfoil_join

    # importa coordenadas de mesh, se quita ultima fila (repetida).
    X = np.copy(mesh.X)[:-1, :]
    Y = np.copy(mesh.Y)[:-1, :]
    # extraer primera columna (perfiles) para eliminar puntos
    # repetidos
    x_perfil        = X[:, 0]
    y_perfil        = Y[:, 0]
    end             = size_flap // 2 + 1 + mesh.airfoil_join \
                        + size_airfoil - 1
    x_perf1         = x_perfil[: end]
    y_perf1         = y_perfil[: end]
    begin           = end + mesh.airfoil_join + 2
    x_perf2         = x_perfil[begin :]
    y_perf2         = y_perfil[begin :]
    x_perfil        = np.concatenate((x_perf1, x_perf2))
    y_perfil        = np.concatenate((y_perf1, y_perf2))
    eta_0           = np.shape(x_perfil)[0]


    # convirtiendo a arreglo 1D
    X               = X[:, 1:]
    Y               = Y[:, 1:]
    X               = X.transpose().flatten()
    Y               = Y.transpose().flatten()
    X               = np.concatenate((x_perfil, X))
    Y               = np.concatenate((y_perfil, Y))

    # se inicia escritura de archivo
    su2_mesh.write('NDIME= 2\n')
    su2_mesh.write('NPOIN= ' + str(NPOIN) + '\n')

    # se escriben las coordenadas de los nodos
    for i in range(NPOIN):
        su2_mesh.write(str(X[i]) + '\t' + str(Y[i]) + '\n')

    # se escriben las celdas y la conectividad entre los nodos que la
    # forman
    NELEM = M_SU2 * N_SU2
    su2_mesh.write('NELEM= ' + str(NELEM) + '\n')

    # primera parte de celdas conectadas al perfil
    size_airfoils   = np.shape(x_perfil)[0]
    end             = size_flap // 2 + 1 + mesh.airfoil_join \
                        + size_airfoil - 2

    for i in range(end):
        su2_mesh.write('9 ' + str(i) + ' ' + str(i+1) + ' '
                       + str(i + size_airfoils + 1) + ' '
                       + str(i + size_airfoils) + '\n')

    # segunda parte, cubre el "regreso" en la O, cubre ultimo pedazo
    # de perfil y la union
    begin           = end
    extrados_flap   = end + 1
    end             += mesh.airfoil_join + 2
    diff            = begin - size_airfoil + 2

    su2_mesh.write('9 ' + str(begin) + ' ' + str(diff) + ' '
                   + str(begin + size_airfoils + 1) + ' '
                   + str(begin + size_airfoils) + '\n')

    begin           += 1

    for i in range(begin, end):
        su2_mesh.write('9 ' + str(diff) + ' ' + str(diff - 1) + ' '
                       + str(i + size_airfoils + 1) + ' '
                       + str(i + size_airfoils) + '\n')
        diff -= 1

    # primer celda extrados flap
    begin = end
    su2_mesh.write('9 ' + str(diff) + ' ' + str(extrados_flap) + ' '
                   + str(begin + size_airfoils + 1) + ' '
                   + str(begin + size_airfoils) + '\n')
    # a partir de este punto todas las celdas siguen la misma
    # secuencia a partir de extrados de flap
    begin += 1
    diff = begin - extrados_flap
    for i in range(begin, M_SU2):
        if i % (M_SU2) != M_SU2 - 1:
            su2_mesh.write('9 ' + str(i - diff) + ' '
                           + str(i - diff + 1) + ' '
                           + str(i + size_airfoils + 1) + ' '
                           + str(i + size_airfoils) + '\n')
        else:
            su2_mesh.write('9 ' + str(i - diff) + ' '
                           + str(i - mesh.M + 2)
                           + ' ' + str(i - diff + 1 ) + ' '
                           + str(i + size_airfoils) + '\n')

    begin = M_SU2
    for i in range(begin, NELEM):
        if i % (M_SU2) != M_SU2 - 1:
            su2_mesh.write('9 ' + str(i - diff) + ' '
                           + str(i - diff + 1) + ' '
                           + str(i + size_airfoils + 1) + ' '
                           + str(i + size_airfoils) + '\n')
        else:
            su2_mesh.write('9 ' + str(i - diff) + ' '
                           + str(i - mesh.M + 2 - diff) + ' '
                           + str(i - diff + 1 ) + ' '
                           + str(i + size_airfoils) + '\n')

    # se escriben las fronteras. Primero FE, luego FI
    NMARK = 3
    su2_mesh.write('NMARK= ' + str(NMARK) + '\n')
    su2_mesh.write('MARKER_TAG= farfield\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(mesh.M - 1) + '\n')
    far0 = NPOIN - M_SU2
    for i in range(far0, NPOIN - 1):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(far0) + '\n')

    # frontera airfoil
    su2_mesh.write('MARKER_TAG= airfoil\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(size_airfoil - 1) + '\n')

    begin = size_flap // 2 + mesh.airfoil_join + 1
    end = begin + size_airfoil - 2
    for i in range(begin, end):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(begin) + '\n')

    # frontera flap
    su2_mesh.write('MARKER_TAG= flap\n')
    su2_mesh.write('MARKER_ELEM= ' + str(size_flap - 1) + '\n')
    for i in range(size_flap // 2):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')

    begin = size_flap // 2 + mesh.airfoil_join + size_airfoil
    end = begin + (size_flap - 1)
    su2_mesh.write('3 ' + str(size_flap // 2) + ' ' + str(begin)
                   + '\n')

    for i in range(begin, eta_0 - 1):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')

    i = eta_0 - 1
    su2_mesh.write('3 ' + str(i) + ' ' + str(0) + '\n')
    su2_mesh.close()

    return


def to_su2_mesh_c_airfoil(mesh, filename):
    '''
    Convierte malla de formato propio a formato de SU2
    Para mallas tipo O
    Con solo un perfil (o cualquier geometria)
    '''

    # guardando X y Y de la malla
    X              = mesh.X
    Y              = mesh.Y
    size_airfoil   = get_size_airfoil(mesh.airfoil_boundary)
    diff           = mesh.M - size_airfoil
    diff           //= 2
    intrados       = np.zeros((diff))
    extrados       = intrados
    is_bound       = np.concatenate([intrados, mesh.airfoil_boundary])
    is_bound       = np.concatenate([is_bound, extrados])

    # extraer primera columna (perfiles) para eliminar puntos
    # repetidos
    x_perfil        = X[:, 0]
    y_perfil        = Y[:, 0]

    x_perfil        = x_perfil[: size_airfoil-1 + diff]
    y_perfil        = y_perfil[: size_airfoil-1 + diff]

    # se elimina j = 0 y se hace arreglo 1D
    X               = X[:, 1:].transpose().flatten()
    Y               = Y[:, 1:].transpose().flatten()

    X               = np.concatenate([x_perfil, X])
    Y               = np.concatenate([y_perfil, Y])

    NPOIN           = np.shape(X)[0]

    # creando archivo
    su2_mesh        = open(filename, 'w')

    # numero de dimensiones
    su2_mesh.write('NDIME= 2\n')

    # lista de nodos
    su2_mesh.write('NPOIN= ' + str(NPOIN) + '\n')
    for i in range(np.shape(X)[0]):
        su2_mesh.write(str(X[i]) + '\t' + str(Y[i]) + '\n')

    # se enlistan las celdas que forman el dominio
    NELEM = (mesh.M - 1) * (mesh.N - 1)
    su2_mesh.write('NELEM= ' + str(NELEM) + '\n')

    # primero de inicio de la malla a final del perfil
    end = diff + size_airfoil - 2
    for i in range(end):
        su2_mesh.write('9 ' + str(i) + ' ' + str(i + 1) + ' '
                       + str(i + end + 2) + ' ' + str(i + end + 1)
                       + '\n')

    # de fin del perfil a regreso de la C en eta = 0
    i += 1
    su2_mesh.write('9 ' + str(i) + ' ' + str(diff) + ' '
                   + str(i + end + 2)
                   + ' ' + str(i + end + 1) + '\n')

    i += 1
    for j in range(diff, 0, -1):
        su2_mesh.write('9 ' + str(j) + ' ' + str(j - 1) + ' '
                       + str(i + end + 2) +  ' ' + str(i + end + 1)
                       + '\n')
        i += 1

    # resto de la malla, a partir de eta = 1
    begin       = mesh.M - 1
    jump        = mesh.M - 3
    i           = 0
    diff_NELEM  = diff
    while begin < NELEM:
        if i == mesh.M - 1:
            diff_NELEM -= 1
            i = 0
        first   = begin - diff_NELEM
        second  = first + 1
        third   = second + mesh.M
        fourth  = first + mesh.M
        su2_mesh.write('9 ' + str(first) + ' ' + str(second) + ' '
                       + str(third) +  ' ' + str(fourth) + '\n')
        i       += 1
        begin   += 1

    # se especifican las fronteras
    su2_mesh.write('NMARK= 2\n')

    # frontera externa
    su2_mesh.write('MARKER_TAG= farfield\n')
    MARKER_ELEMS = mesh.M - 1 + (mesh.N - 1) * 2
    su2_mesh.write('MARKER_ELEMS= ' + str(MARKER_ELEMS) + '\n')

    # parte inferior de C
    begin = diff + size_airfoil - 1
    su2_mesh.write('3 ' + str(0) + ' ' + str(begin) + '\n')
    for i in range(mesh.N - 2):
        end     = begin + mesh.M
        su2_mesh.write('3 ' + str(begin) + ' ' + str(end) + '\n')
        begin   = end

    # parte C
    end += mesh.M - 1
    for i in range(begin, end):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')

    # parte superior de C
    begin = end
    for i in range(mesh.N - 2):
        end     = begin - mesh.M
        su2_mesh.write('3 ' + str(begin) + ' ' + str(end) + '\n')
        begin   = end
    su2_mesh.write('3 ' + str(begin) + ' ' + str(0) + '\n')

    # frontera interna
    MARKER_ELEMS    = size_airfoil - 1
    begin           = diff
    end             = diff + size_airfoil - 2
    su2_mesh.write('MARKER_TAG= airfoil\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(MARKER_ELEMS) + '\n')
    for i in range(begin, end):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    su2_mesh.write('3 ' + str(i + 1) + ' ' + str(begin) + '\n')
    su2_mesh.close()

    return

def to_su2_mesh_c_airfoil_n_flap(mesh, filename):
    '''
    Convierte malla de formato propio a formato de SU2
    Para mallas tipo C
    Para perfiles con external airfoil flap (o en general,
        2 geometrias separadas)
    '''

    size_airfoil, size_flap = get_size_airfoil_n_flap(\
                                        mesh.airfoil_boundary[:-1])

    union           = mesh.airfoil_join
    diff            = mesh.M - (size_airfoil + size_flap + 1
                                + union * 2)
    diff            //= 2

    # creando archivo de malla para SU2
    su2_mesh        = open(filename, 'w')

    M_SU2           = mesh.M - 1
    N_SU2           = mesh.N - 1
    NPOIN           = size_airfoil - 1 + size_flap - 1 + union + diff
    NPOIN           += mesh.M * (N_SU2)
    NELEM           = M_SU2 * N_SU2

    # importa coordenadas de mesh, se quita ultima fila (repetida).
    X               = np.copy(mesh.X)
    Y               = np.copy(mesh.Y)

    # extraer primera columna (perfiles) para eliminar puntos
    # repetidos
    x_perfil        = X[:, 0]
    y_perfil        = Y[:, 0]
    end             = diff + size_flap // 2 + 1 + mesh.airfoil_join \
                        + size_airfoil - 1
    x_perf1         = x_perfil[: end]
    y_perf1         = y_perfil[: end]
    begin           = end + mesh.airfoil_join + 2
    end             = -diff - 1
    x_perf2         = x_perfil[begin : end]
    y_perf2         = y_perfil[begin : end]
    x_perfil        = np.concatenate((x_perf1, x_perf2))
    y_perfil        = np.concatenate((y_perf1, y_perf2))
    eta_0           = np.shape(x_perfil)[0]

    # convirtiendo a arreglo 1D
    X               = X[:, 1:]
    Y               = Y[:, 1:]
    X               = X.transpose().flatten()
    Y               = Y.transpose().flatten()
    X               = np.concatenate((x_perfil, X))
    Y               = np.concatenate((y_perfil, Y))

    # numero de dimensiones
    su2_mesh.write('NDIME= 2\n')

    # lista de nodos
    su2_mesh.write('NPOIN= ' + str(NPOIN) + '\n')
    for i in range(NPOIN):
        su2_mesh.write(str(X[i]) + '\t' + str(Y[i]) + '\n')

    # se enlistan las celdas que forman el dominio
    su2_mesh.write('NELEM= ' + str(NELEM) + '\n')

    # primero de inicio de la malla a final del perfil
    end = diff + size_flap // 2 + 1 + union + size_airfoil - 2
    for i in range(end):
        su2_mesh.write('9 ' + str(i) + ' ' + str(i + 1) + ' '
                       + str(i + eta_0 + 1) + ' ' + str(i + eta_0)
                       + '\n')

    # de fin del perfil a inicio de union
    i += 1
    su2_mesh.write('9 ' + str(i) + ' ' + str(i - size_airfoil + 2)
                   + ' ' + str(i + eta_0 + 1) + ' ' + str(i + eta_0)
                   + '\n')

    # union de regreso
    begin = i - size_airfoil + 2
    for j in range(begin, begin - union - 1, -1):
        i += 1
        su2_mesh.write('9 ' + str(j) + ' ' + str(j -1) + ' '
                       + str(i + eta_0 + 1) + ' ' + str(i + eta_0)
                       + '\n')

    # primer celda del borde de ataque de flap. regreso
    i   += 1
    j   -= 1
    end += 1
    su2_mesh.write('9 ' + str(j) + ' ' + str(end) + ' '
                   + str(i + eta_0 + 1) + ' ' + str(i + eta_0) + '\n')

    # resto del flap
    begin   = end
    end     = begin + size_airfoil // 2 - 2
    for j in range(begin, end):
        i += 1
        su2_mesh.write('9 ' + str(j) + ' ' + str(j + 1) + ' '
                       + str(i + eta_0 + 1) +  ' ' + str(i + eta_0)
                       + '\n')

    # ultima celda borde de salida flap
    i += 1
    su2_mesh.write('9 ' + str(eta_0 - 1) + ' ' + str(diff) + ' '
                    + str(i + eta_0 + 1) + ' ' + str(i + eta_0)
                   + '\n')

    # ultima seccion de regreso de la C
    for j in range (diff, 0, -1):
        i += 1
        su2_mesh.write('9 ' + str(j) + ' ' + str(j - 1) + ' '
                       + str(i + eta_0 + 1) + ' ' + str(i + eta_0)
                       + '\n')

    # resto de la malla, a partir de eta = 1
    begin   = M_SU2
    jump    = mesh.M - 3
    i       = 0
    diff_NELEM = begin - eta_0
    while begin < NELEM:
        if i == mesh.M - 1:
            diff_NELEM -= 1
            i = 0
        first   = begin - diff_NELEM
        second  = first + 1
        third   = second + mesh.M
        fourth  = first + mesh.M
        su2_mesh.write('9 ' + str(first) + ' ' + str(second) + ' '
                       + str(third) +  ' ' + str(fourth) + '\n')
        i += 1
        begin += 1

    # se especifican las fronteras
    su2_mesh.write('NMARK= 3\n')

    # frontera externa
    su2_mesh.write('MARKER_TAG= farfield\n')
    MARKER_ELEMS = M_SU2 + N_SU2 * 2
    su2_mesh.write('MARKER_ELEMS= ' + str(MARKER_ELEMS) + '\n')

    # parte inferior de C
    begin = eta_0
    su2_mesh.write('3 ' + str(0) + ' ' + str(begin) + '\n')
    for i in range(mesh.N - 2):
        end = begin + mesh.M
        su2_mesh.write('3 ' + str(begin) + ' ' + str(end) + '\n')
        begin = end

    # parte de la C en sentido horario
    begin = NPOIN - mesh.M
    for i in range(begin, NPOIN - 1):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')

    begin   = NPOIN - 1
    end     = begin - mesh.M
    for i in range(N_SU2 - 1):
        su2_mesh.write('3 ' + str(begin) + ' ' + str(end) + '\n')
        begin   = end
        end     -= mesh.M
    su2_mesh.write('3 ' + str(begin) + ' ' + str(0) + '\n')

    # airfoil
    MARKER_ELEMS    = size_airfoil - 1
    begin           = diff + size_flap // 2 + 1 + union
    end             = begin + size_airfoil - 2
    su2_mesh.write('MARKER_TAG= airfoil\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(MARKER_ELEMS) + '\n')
    for i in range(begin, end):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')
    i += 1
    su2_mesh.write('3 ' + str(i) + ' ' + str(begin) + '\n')
    end_airfoil = i

    # flap
    MARKER_ELEMS    = size_flap - 1
    su2_mesh.write('MARKER_TAG= flap\n')
    su2_mesh.write('MARKER_ELEMS= ' + str(MARKER_ELEMS) + '\n')

    # intrados hasta borde de ataque
    begin           = diff
    end             = begin + size_flap // 2
    for i in range(begin, end):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')

    # primer elemento borde de ataque extrados
    i           += 1
    end_airfoil += 1
    su2_mesh.write('3 ' + str(i) + ' ' + str(end_airfoil) + '\n')

    # resto de extrados. regreso
    begin   = end_airfoil
    end     = begin +  size_flap // 2 - 2
    i       = begin
    for i in range(begin, end):
        su2_mesh.write('3 ' + str(i) + ' ' + str(i + 1) + '\n')

    su2_mesh.write('3 ' + str(eta_0 - 1) + ' ' + str(diff) + '\n')
    su2_mesh.close()

    return

