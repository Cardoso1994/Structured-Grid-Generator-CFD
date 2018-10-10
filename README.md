# Mesh Generator
Mesh generator for O and C grid types.
Generates grid by algebraic, and partial differential equations (elliptic, hyperbolic and parabolic)

## Description
This mesh generator is intended to generate O and C grids for computational fluid dynamics applications.  
It has a module called `airfoil` for importing any airfoil provided the user has the point distribution.  
Some airfoil data can be found at http://airfoiltools.com/  
On the other hand, I've created a method for generating NACA airfoils of the 4th series. The user just needs to specify the number of points the airfoil will be created with.  
  
  
About the grid generation, there are 3 main files:  
  *mesh.py  
  *mesh_o.py  
  *mesh_c.py  

## Contact Me
If you found a bug, or have any suggestions on how the performance of the generator might be improved, please contact me at:  
marcoacardosom@gmail.com
