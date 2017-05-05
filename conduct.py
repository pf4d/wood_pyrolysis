from fenics import *
import numpy as np

# furnace dimensions :
x  = 1.0
y  = 0.6
z  = 1.0

# square element dimensions :
s  = 0.05

# wood dimensions :
wx = 0.1
wy = 0.1
wz = 0.025

# number of elements :
nx = int(round(x / s))
ny = int(round(y / s))
nz = int(round(z / s))

# finite-element mesh :
p0   = Point( -x/2, -y/2, -z/2)
p1   = Point(  x/2,  y/2,  z/2)
mesh = BoxMesh(p0, p1, nx, ny, nz) 

for i in range(5):
  cell_markers = CellFunction("bool", mesh)
  cell_markers.set_all(False)
  for cell in cells(mesh):
    c_x = cell.midpoint().x()
    c_y = cell.midpoint().y()
    c_z = cell.midpoint().z()
    if c_x > -wx/2 and c_x < wx/2 and \
       c_y > -wy/2 and c_y < wy/2 and \
       c_z > -wz/2 and c_z < wz/2:
      cell_markers[cell] = True
    v_c = cell.get_vertex_coordinates().reshape((-1,4))
    v_max = np.max(v_c, axis=1)
    v_min = np.min(v_c, axis=1)
    if (v_min[0] >= -wx/2 and v_max[0] <= wx/2) and \
       (v_min[1] >= -wy/2 and v_max[1] <= wy/2) and \
       (v_min[2] >= -wz/2 and v_max[2] <= wz/2):
      cell_markers[cell] = True
  mesh = refine(mesh, cell_markers)

# function space :
Qe   = FiniteElement('CG', 'tetrahedron', 1)
Q    = FunctionSpace(mesh, Qe)

# general test for solid or vapor domain :
w_x  = "x[0] > -wx/2 && x[0] < wx/2"
w_y  = "x[1] > -wy/2 && x[1] < wy/2"
w_z  = "x[2] > -wz/2 && x[2] < wz/2"
eva  = "? v1 : v2 "
code = w_x + "&&" + w_y + "&&" + w_z + eva

omega_s = Expression(code, wx=wx, wy=wy, wz=wz, v1=1, v2=0, element=Qe)
omega_v = Expression(code, wx=wx, wy=wy, wz=wz, v1=0, v2=1, element=Qe)

# wood properties (fir) :
rho_s = 363.0
k_s   = 0.169 
c_p_s = 1500.0

# gas properties (nitrogen) :
rho_v = 1.251
k_v   = 0.026
c_p_v = 1040.0
gam_v = 1.4
c_v_v = c_p_v / gam_v

File('omega_s.pvd') << interpolate(omega_s, Q)
File('omega_v.pvd') << interpolate(omega_v, Q)
