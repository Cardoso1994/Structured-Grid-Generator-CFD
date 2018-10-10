# Mesh Generator
Mesh generator for O and C grid types.
Generates grid by algebraic, and partial differential equations (elliptic, hyperbolic and parabolic)

## Description
This mesh generator is intended to generate O and C grids for computational fluid dynamics applications.  
It has a module called `airfoil` for importing any airfoil provided the user has the point distribution.  
Some airfoil data can be found at http://airfoiltools.com/  
On the other hand, I've created a method for generating NACA airfoils of the 4th series. The use just needs to specify the number of points the airfoil will be created with.
