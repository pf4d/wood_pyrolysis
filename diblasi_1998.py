from fenics import *

# constants :
R          = 8.3144      # universal gas constant
sigma      = 5.67e-8     # Boltzman's constant
g_a        = 9.80665     # gravitational acceleration

# pre-exponential factor for wood (W), tar (r) and char + gas (av) :
A_W        = 2.8e19
A_av       = 1.3e10
A_r        = 3.28e14

# activation energy for wood (W), tar (r) and char + gas (av) :
E_W        = 242.4e3
E_av       = 150.5e3
E_r        = 196.5e3

nu_G       = 0.65
nu_C       = 0.35
delta_h_W  = 0.0
delta_h_av = -418.0e3
delta_h_r  = -418.0e3
c_W        = 2.3e3
c_A        = 2.3e3
c_g        = 1.8e3
c_C        = 1.1e3
c_r        = 1.1e3      # FIXME: missing for tar
k_g        = 25.77e-3
mu         = 3e-5       # gas viscosity
omega      = 1.0        # emissiviy
h_c        = 20         # convective heat transfer coefficent

# pore diameter :
d_x        = 4e-5
d_y        = 4e-5
d          = as_vector([d_x,   d_y  ])

# permeability :
B_W_x      = 1e-14
B_A_x      = 1e-14
B_C_x      = 5e-12
B_W_y      = 1e-11
B_A_y      = 1e-11
B_C_y      = 5e-11
B_W        = as_vector([B_W_x, B_W_y])
B_A        = as_vector([B_A_x, B_A_y])
B_C        = as_vector([B_C_x, B_C_y])

# thermal conductivity :
k_W_x      = 10.5e-2
k_A_x      = 10.5e-2
k_C_x      = 7.1e-2
k_W_y      = 25.5e-2
k_A_y      = 25.5e-2
k_C_y      = 10.46e-2
k_W        = as_vector([k_W_x, k_W_y])
k_A        = as_vector([k_A_x, k_A_y])
k_C        = as_vector([k_C_x, k_C_y])

# gravitational acceleration vector :
g          = as_vector([0.0, -g_a])

# initial conditions :
T_0        = 300.0
rho_W_0    = 400.0
rho_A_0    = 0.0
rho_C_0    = 0.0
V_S_0      = 1.0
p_0        = 1e5
W_g        = 1.0  # FIXME:need an expression for the molecular weight of gas
rho_g_0    = p_0 * W_g / (R * T_0)

# boundary conditions :
T_inf      = 900.0                      # ambient temperature
p_inf      = p_0                        # ambient gas pressure

# the gas density boundary Dirichlet boundary condition :
rho_g_inf = Expression('p_inf * W_g / (R * Tp)', \
                       p_inf=p_inf, W_g=W_g, R=R, Tp=T_0, degree=2)

# mesh varaiables :
tau       = 1.0#0.5e-2                       # width of domain
dn        = 32                           # number of elements


# volume of solid :
def V_S(rho_W, rho_C, rho_A):
  return (rho_W + rho_C + rho_A) * V_S_0 / rho_W_0

# reaction-rate factor :
def K(T, A, E):
  return A * exp( - E / RT)

# virgin wood reaction rate :
def K_W(T):
  return K(T, A_W, E_W)

# char and gas reaction rate :
def K_av(T):
  return K(T, A_av, E_av)

# tar reaction rate :
def K_r(T):
  return K(T, A_r, E_r)

# porosity :
def epsilon(rho_W, rho_C, rho_A, V):
  return (V - V_S(rho_W, rho_C, rho_A)) / V

# ratio of current solid mass to initial solid mass :
def eta(rho_W, rho_A):
  return (rho_A + rho_W) / rho_W_0

# thermal conductivity vector :
def k(rho_W, rho_C, rho_A, T, V):
  k_v = + eta(rho_W, rho_A) * k_W \
        + (1 - eta(rho_W, rho_A)) * k_C \
        + epsilon(rho_W, rho_C, rho_A, V) * k_g \
        + sigma * T**3 * d / omega
  return k_v

# permeability vector :
def B(rho_W, rho_A):
  B_v = + eta(rho_W, rho_A) * B_W \
        + (1 - eta(rho_W, rho_A)) * B_C
  return B_v

# gas pressure :
def p(rho_g, T):
  return rho_g * R * T / W_g

# gas velocity from Darcy's law :
def U(rho_W, rho_A, rho_g, T):
  return - B(rho_W, rho_A) / mu * ( grad(p(rho_g, T) + rho_g*g) )

# gas mass flux :
def j_rho_g(rho_W, rho_A, rho_g, T):
  return rho_g * U(rho_W, rho_A, rho_g, T)

# advective temperature flux :
def j_a_T(rho_W, rho_A, rho_g, T):
  return c_g * j_rho_g(rho_W, rho_A, rho_g, T) * grad(T)

# diffusive temperature flux :
def j_d_T(rho_W, rho_C, rho_A, T, V):
  return k(rho_W, rho_C, rho_A, T, V) * grad(T)

# total temperature flux :
def j_T(rho_W, rho_A, rho_g, T, V):
  return j_a_T(rho_W, rho_A, rho_g, T) + j_d_T(rho_W, rho_A, T, V)

# enthalpy variation due to chemical reactions :
def q_r(rho_W, rho_A, T):
  q_r_v = + K_W(T)*rho_W*(delta_h_W + (T - T_0)*(c_W - c_A)) \
          + K_av(T)*rho_A*(delta_h_av + (T - T_0)*(c_A - nu_C*c_C - nu_g*c_G)) \
          + K_r(T)*rho_A*(delta_h_r + (T - T_0)*(c_A - c_r))
  return q_r_v

# temperature flux boundary condition :
def k_grad_T(T):
  k_grad_T_v = - omega * sigma * (T**4 - T_inf**4) - h_c * (T - T_inf)
  return as_vector([k_grad_T_v, k_grad_T_v])

# create a genreic box mesh, we'll fit it to geometry below :
p1    = Point(0.0, 0.0)                  # origin
p2    = Point(tau, tau)                  # x, y corner 
mesh  = RectangleMesh(p1, p2, dn, dn)    # a box to fill the void 

# Define finite elements spaces and build mixed space
BDM   = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG    = FiniteElement("DG",  mesh.ufl_cell(), 0)
CG    = FiniteElement("CG",  mesh.ufl_cell(), 1)
We    = MixedElement([BDM, DG, CG])
W     = FunctionSpace(mesh, BDM * DG)

# Define trial and test functions
j,   u    = TrialFunctions(W)
phi, psi  = TestFunctions(W)

f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",
               degree=2)

# Define variational form
delta_j = dot(j, phi) * dx - div(phi) * u * dx
delta_u = div(j) * psi * dx + f*psi*dx
delta   = delta_j + delta_u
L       = rhs(delta)
a       = lhs(delta)

# Define function G such that G \cdot n = g
class BoundarySource(Expression):
  def __init__(self, mesh, **kwargs):
    self.mesh = mesh
  def eval_cell(self, values, x, ufc_cell):
    cell      = Cell(self.mesh, ufc_cell.index)
    n         = cell.normal(ufc_cell.local_facet)
    g         = sin(5*x[0])
    values[0] = g*n[0]
    values[1] = g*n[1]
  def value_shape(self):
    return (2,)

G = BoundarySource(mesh, degree=2)

# Define essential boundary
def boundary(x):
  return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

bc = DirichletBC(W.sub(0), G, boundary)

# Compute solution
w   = Function(W)
solve(a == L, w, bc)
j,u = w.split()

# Plot sigma and u
plot(j)
plot(u)
interactive()



