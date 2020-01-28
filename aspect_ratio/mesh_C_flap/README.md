# Análisis de mallas tipo C
Perfil NACA 2412 con flap 2412

dx = 0.055
dy = 0.05
union = 6
c_flap = 0.2 c_perfil
alpha_flap = 15

## Método Laplace
* configuración central 53 x 75 [141 x 75]
* Aumento en xi, 61 x 75 [153 x 75]
* Aumento en eta, 53 x 95 [103 x 95]
* omega = 1.4

## Método Poisson
* mismas dimensiones de malla que en Laplace
* aa = 26.5
* cc = 6.1
* omega = 1.5


### Notas
Tarda considerablemente más en converger, con el aumento en eta que en xi, para
una configuración inicial
