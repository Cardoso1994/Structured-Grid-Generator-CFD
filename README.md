# Mesh Generator
Mesh generator for O and C grid types. Generates grid by algebraic, and partial
differential equations (elliptic, hyperbolic and parabolic)

## Description
This mesh generator is intended to generate O and C grids for computational
fluid dynamics applications. It has a module called `airfoil` for importing any
airfoil provided the user has the point distribution. Some airfoil data can be
found at [Airfoil Tools](http://airfoiltools.com/). On the other hand, I've
created a method for generating NACA airfoils of the 4th series. The user just
needs to specify the number of points the airfoil will be created with.

About the grid generation, there are 3 main files:
* `mesh.py`
* `mesh_o.py`
* `mesh_c.py`

In `mesh.py` there characteristic of the mesh are defined, for example the grid
dimensions `M x N`. In this file the are defined the algebraic methods as it
makes no difference what type of mesh we are working with. There's also an
implementation of a `plot` method, that as the name suggests plots the
generated grid.

In `mesh_c.py` and `mesh_o.py` are implemented all the methods that need
specific treatment depending on which grid the user wants to generate. You can
find there methods for Elliptic, Parabolic and Hyperbolic methods. In case of
elliptic scheme, the user has a method for Laplace's Equation and another
method for Poisson Equation, and for both of them the user has the capability
of using one of three iterative methods:
* Jacobi
* Gauss - Seidel
* SOR (in this one the user must specify the value of `omega`)


## Contact Me
If you found a bug, or have any suggestions on how the performance of the
generator might be improved, please contact me at:
marcoacardosom@gmail.com
