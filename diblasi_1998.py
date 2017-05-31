from fenics import *
from time   import time

# constants :
R          = 8.3144      # universal gas constant
sigma      = 5.67e-8     # Boltzman's constant
g_a        = 9.80665     # gravitational acceleration

# pre-exponential factor for wood (W), tar (r) and char + gas (av) :
A_W        = 2.8e19
A_C        = 1.3e10
A_r        = 3.28e14

# activation energy for wood (W), tar (r) and char + gas (av) :
E_W        = 242.4e3
E_C        = 150.5e3
E_r        = 196.5e3

nu_G       = 0.65
nu_C       = 0.35
delta_h_W  = 0.0
delta_h_C  = -418.0e3
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
W_g        = 2.897e-2  # FIXME: need molecular weight of gas
rho_g_0    = p_0 * W_g / (R * T_0)

# time parameters :
dt         = 0.1         # time step
t0         = 0.0         # start time
t          = t0          # current time
tf         = 60.0*60.0   # final time

# boundary conditions :
T_inf      = 900.0                      # ambient temperature
p_inf      = p_0                        # ambient gas pressure

# the gas density boundary Dirichlet boundary condition :
rho_g_inf  = Expression('p_inf * W_g / (R * Tp)', \
                        p_inf=p_inf, W_g=W_g, R=R, Tp=T_0, degree=2)

# mesh varaiables :
tau        = 1.0e-2                     # width of domain
dn         = 32                         # number of elements


# volume of solid :
def V_S(rho_W, rho_C, rho_A):
  return (rho_W + rho_C + rho_A) * V_S_0 / rho_W_0

# reaction-rate factor :
def K(T, A, E):
  return A * exp( - E / RT)

# virgin wood reaction rate factor :
def K_W(T):
  return K(T, A_W, E_W)

# char and gas reaction rate factor :
def K_C(T):
  return K(T, A_C, E_C)

# tar reaction rate factor :
def K_r(T):
  return K(T, A_r, E_r)

# virgin wood reaction rate :
def r_W(rho_W, T):
  return K_W(T) * rho_W

# char and gas reaction rate :
def r_C(rho_A, T):
  return K_C(T) * rho_A

# tar reaction rate :
def r_r(rho_A, T):
  return K_r(T) * rho_A

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
          + K_C(T)*rho_A*(delta_h_C + (T - T_0)*(c_A - nu_C*c_C - nu_g*c_G)) \
          + K_r(T)*rho_A*(delta_h_r + (T - T_0)*(c_A - c_r))
  return q_r_v

# temperature flux boundary condition :
def k_grad_T(T):
  k_grad_T_v = - omega * sigma * (T**4 - T_inf**4) - h_c * (T - T_inf)
  return as_vector([k_grad_T_v, k_grad_T_v])

# time derivative :
def dudt(u,u1):  return (u - u1) / dt

# create a mesh :
p1    = Point(0.0, 0.0)                  # origin
p2    = Point(tau, tau)                  # x, y corner 
mesh  = RectangleMesh(p1, p2, dn, dn)    # a box to fill the void 

# define finite elements spaces and build mixed space :
BDM   = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG    = FiniteElement("DG",  mesh.ufl_cell(), 0)
CG    = FiniteElement("CG",  mesh.ufl_cell(), 1)
We    = MixedElement([BDM, DG, CG, CG, CG, CG])
W     = FunctionSpace(mesh, We)

# outward-facing normal vector :
n     = FacetNormal(mesh)

# define trial and test functions :
dW        = TrialFunction(W)
Phi       = TestFunction(W)
w         = Function(W)
w1        = Function(W)

# get the individual functions :
phi,   psi,    xi,     chi,    zeta,   beta  = Phi.split()
j,     rho_g,  rho_W,  rho_r,  rho_C,  T     = w.split()
j1,    rho_g1, rho_W1, rho_r1, rho_C1, T     = w1.split()

# gas mass flux residual :
j_p           = - rho_g * B(rho_W, rho_A) / mu * p(rho_g, T)
j_p_inf       = - rho_g_inf * B(rho_W, rho_A) / mu * p(rho_g_inf, T_inf)
j_g           = - rho_g * B(rho_W, rho_A) / mu * rho_g*g
delta_j_rho_g = + dot(j, phi) * dx \
                + div(phi) * j_p * dx \
                - j_p_inf * dot(phi, n) * ds \
                - dot(j_g, phi) * dx

# gas mass balance residual :
ep1           = epsilon(rho_W1, rho_C1, rho_A1, V)
ep            = epsilon(rho_W,  rho_C,  rho_A,  V)
delta_rho_g   = + dudt(ep*rho_g, ep1*rho_g1) * psi * dx \
                + div(j) * psi * dx \
                - (nu_G*r_C(rho_A, T) + r_t(rho_A, T))*psi*dx

# virgin solid wood mass balance :
delta_rho_W   = + dudt(rho_W, rho_W1) * xi * dx \
                + r_W(rho_W, T) * xi * dx

# active intermediate solid wood (tar) mass balance :
delta_rho_r   = + dudt(rho_A, rho_A1) * chi * dx \
                + (r_C(rho_A, T) + r_t(rho_A, T) - r_W(rho_W, T)) * chi * dx

# solid char mass balance :
delta_rho_C   = + dudt(rho_C, rho_C1) * zeta * dx \
                - nu_C * r_C(rho_A, T) * zeta * dx

# enthalpy balance :
T_factor      = (rho_C*c_C + rho_W*c_W + rho_A*c_A + ep*rho_g*c_g)
delta_T       = + T_factor * dudt(T, T1) * beta * dx \
                + c_g*dot(j, grad(T)) * beta * dx \
                + k(rho_W, rho_C, rho_A, T, V) * dot(grad(T), grad(beta)) * dx \
                - dot(k_grad_T(T), n) * beta * dx \
                - q_r(rho_W, rho_A, T)

# total residual :
delta         = + delta_j_rho_g + delta_rho_g + delta_rho_W \
                + delta_rho_r + delta_rho_C + delta_T

# Jacobian :
J             = derivative(delta, W, dW)

# define essential boundary :
def boundary(x, on_boundary
  return on_boundary 

bc = DirichletBC(W.sub(0), G, boundary) #FIXME

# start the timer :
start_time = time()

# loop over all times :
while t < tf:

  # start the timer :
  tic = time()

  # Compute solution
  solve(delta == 0, w, J=J, bcs=bc)
  j,u = w.split()
  
  # increment time step :
  s = '>>> Time: %g s, CPU time for last dt: %.3f s <<<'
  print_text(s % (t, time()-tic), 'red', 1)

  t += dt

# calculate total time to compute
sec = time() - start_time
mnn = sec / 60.0
hor = mnn / 60.0
sec = sec % 60
mnn = mnn % 60
text = "total time to perform transient run: %02d:%02d:%02d" % (hor,mnn,sec)
print_text(text, 'red', 1)

# Plot sigma and u
plot(j)
plot(u)
interactive()



