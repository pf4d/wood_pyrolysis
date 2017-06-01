from fenics import *
from time   import time
from helper import print_text, print_min_max

parameters['form_compiler']['quadrature_degree'] = 2

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

nu_g       = 0.65
nu_C       = 0.35
delta_h_W  = 0.0
delta_h_C  = -418.0e3
delta_h_r  = -418.0e3
c_W        = 2.3e3
c_A        = 2.3e3
c_g        = 1.8e3
c_C        = 1.1e3
c_r        = 1.1e3      # FIXME: missing for tar
mu         = 3e-5       # gas viscosity
omega      = 1.0        # emissiviy
h_c        = 20         # convective heat transfer coefficent

# pore diameter :
d_x        = 4e-5
d_y        = 4e-5
d          = as_vector([d_x,   d_y  ])

# permeability :
B_W_x      = 1e-14
#B_W_y      = 1e-14
B_W_y      = 1e-11
B_W        = as_vector([B_W_x, B_W_y])

B_A_x      = 1e-14
#B_A_y      = 1e-14
B_A_y      = 1e-11
B_A        = as_vector([B_A_x, B_A_y])

B_C_x      = 5e-12
#B_C_y      = 5e-12
B_C_y      = 5e-11
B_C        = as_vector([B_C_x, B_C_y])

# thermal conductivity :
k_W_x      = 10.5e-2
#k_W_y      = 10.5e-2
k_W_y      = 25.5e-2
k_W        = as_vector([k_W_x, k_W_y])

k_A_x      = 10.5e-2
#k_A_y      = 10.5e-2
k_A_y      = 25.5e-2
k_A        = as_vector([k_A_x, k_A_y])

k_C_x      = 7.1e-2
#k_C_y      = 7.1e-2
k_C_y      = 10.46e-2
k_C        = as_vector([k_C_x, k_C_y])

k_g_x      = 25.77e-3
k_g_y      = 25.77e-3
k_g        = as_vector([k_g_x, k_g_y])

# gravitational acceleration vector :
g          = as_vector([0.0, -g_a])

# initial conditions :
T_0        = 300.0
rho_W_0    = 400.0
rho_A_0    = 0.0
rho_C_0    = 0.0
V_S_0      = 0.4
p_0        = 1e5
W_g        = 2.897e-2  # FIXME: need molecular weight of gas
rho_g_0    = p_0 * W_g / (R * T_0)

# time parameters :
dt         = 0.01                       # time step
t0         = 0.0                        # start time
t          = t0                         # current time
tf         = 20                         # final time

# boundary conditions :
T_inf      = 900.0                      # ambient temperature
p_inf      = p_0                        # ambient gas pressure

# the gas density boundary Dirichlet boundary condition :
rho_g_inf  = Expression('p_inf * W_g / (R * Tp)', \
                        p_inf=p_inf, W_g=W_g, R=R, Tp=T_0, degree=2)

# mesh varaiables :
tau        = 1.0e-2                     # width of domain
dn         = 32                        # number of elements


# volume of solid :
def V_S(rho_W, rho_C, rho_A):
  return (rho_W + rho_C + rho_A) * V_S_0 / rho_W_0

# reaction-rate factor :
def K(T, A, E):
  return A * exp( - E / (R * T) )

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
          + K_C(T)*rho_A*(delta_h_C + (T - T_0)*(c_A - nu_C*c_C - nu_g*c_g)) \
          + K_r(T)*rho_A*(delta_h_r + (T - T_0)*(c_A - c_r))
  return q_r_v

# temperature flux boundary condition :
def kdTdn(T):
  return - omega * sigma * (T**4 - T_inf**4) - h_c * (T - T_inf)

# time derivative :
def dudt(u,u1):  return (u - u1) / dt

# create a mesh :
p1     = Point(0.0, 0.0)                  # origin
p2     = Point(tau, tau)                  # x, y corner 
mesh   = RectangleMesh(p1, p2, dn, dn)    # a box to fill the void 

# define finite elements spaces and build mixed space :
BDMe   = FiniteElement("BDM", mesh.ufl_cell(), 1)
DGe    = FiniteElement("DG",  mesh.ufl_cell(), 0)
CGe    = FiniteElement("CG",  mesh.ufl_cell(), 1)
BDM    = FunctionSpace(mesh, BDMe)
DG     = FunctionSpace(mesh, DGe)
CG     = FunctionSpace(mesh, CGe)
VCG    = VectorFunctionSpace(mesh, 'CG', 1)
We     = MixedElement([BDMe, DGe, DGe, DGe, DGe, CGe])
W      = FunctionSpace(mesh, We)

def boundary(x, on_boundary):
  return on_boundary

bc = DirichletBC(W.sub(5), 600, boundary)

# outward-facing normal vector :
n      = FacetNormal(mesh)
h      = CellSize(mesh)
V      = Constant(1.0)#CellVolume(mesh)

# define trial and test functions :
Phi    = TestFunction(W)
dw     = TrialFunction(W)
w      = Function(W)
w1     = Function(W)

# get the individual functions :
phi_x,   phi_y,    psi,    xi,     chi,    zeta,   beta  = Phi
j_g_x,   j_g_y,    rho_g,  rho_W,  rho_A,  rho_C,  T     = w
j1_g_x,  j1_g_y,   rho_g1, rho_W1, rho_A1, rho_C1, T1    = w1

phi    = as_vector([phi_x,  phi_y ])
j_g    = as_vector([j_g_x,  j_g_y ])
j1_g   = as_vector([j1_g_x, j1_g_y])

# set initial conditions:
ji     = interpolate(Constant((0.0,0.0)), BDM)
rho_gi = interpolate(Constant(rho_g_0),   DG)
rho_Wi = interpolate(Constant(rho_W_0),   DG)
rho_Ai = interpolate(Constant(rho_A_0),   DG)
rho_Ci = interpolate(Constant(rho_C_0),   DG)
Ti     = interpolate(Constant(T_0),       CG)

# assign initial values :
assign(w,  [ji, rho_gi, rho_Wi, rho_Ai, rho_Ci, Ti])

# pressure gradient flow integrated by parts :
j_g_p_x       = - rho_g * B(rho_W, rho_A)[0] / mu * p(rho_g, T)
j_g_p_y       = - rho_g * B(rho_W, rho_A)[1] / mu * p(rho_g, T)

# boundary gas flux :
j_g_p_inf_x   = - rho_g_inf * B(rho_W, rho_A)[0] / mu * p(rho_g_inf, T)
j_g_p_inf_y   = - rho_g_inf * B(rho_W, rho_A)[1] / mu * p(rho_g_inf, T)

# gravitational flow :
j_g_g_x       = - rho_g * B(rho_W, rho_A)[0] / mu * rho_g*g[0]
j_g_g_y       = - rho_g * B(rho_W, rho_A)[1] / mu * rho_g*g[1]
j_g_g         = as_vector([j_g_g_x, j_g_g_y])

# gas mass flux residual :
delta_j_rho_g = + dot(j_g, phi) * dx \
                + j_g_p_x * phi[0].dx(0) * dx \
                + j_g_p_y * phi[1].dx(1) * dx \
                - j_g_p_inf_x * phi[0] * n[0] * ds \
                - j_g_p_inf_y * phi[1] * n[1] * ds \
#                - dot(j_g_g, phi) * dx

# gas mass balance residual :
ep1           = epsilon(rho_W1, rho_C1, rho_A1, V)
ep            = epsilon(rho_W,  rho_C,  rho_A,  V)
delta_rho_g   = + dudt(ep*rho_g, ep1*rho_g1) * psi * dx \
                + div(j_g) * psi * dx \
                - (nu_g*r_C(rho_A, T) + r_r(rho_A, T))*psi*dx

# virgin solid wood mass balance :
delta_rho_W   = + dudt(rho_W, rho_W1) * xi * dx \
                + r_W(rho_W, T) * xi * dx

# active intermediate solid wood (tar) mass balance :
delta_rho_A   = + dudt(rho_A, rho_A1) * chi * dx \
                + (r_C(rho_A, T) + r_r(rho_A, T) - r_W(rho_W, T)) * chi * dx

# solid char mass balance :
delta_rho_C   = + dudt(rho_C, rho_C1) * zeta * dx \
                - nu_C * r_C(rho_A, T) * zeta * dx
 
# intrinsic time parameter :
def tau(u, v, k):
  order = 1

  # the Peclet number : 
  Unorm  = sqrt(dot(v, v) + DOLFIN_EPS)
  knorm  = sqrt(dot(k, k) + DOLFIN_EPS)
  PE     = Unorm * h / (2*knorm)

  # for linear elements :
  if order == 1:
    xi     = 1/tanh(PE) - 1/PE

  # for quadradic elements :
  if order == 2:
    xi_1  = 0.5*(1/tanh(PE) - 2/PE)
    xi    =     ((3 + 3*PE*xi_1)*tanh(PE) - (3*PE + PE**2*xi_1)) \
             /  ((2 - 3*xi_1*tanh(PE))*PE**2)
  
  # intrinsic time parameter :
  tau_n = h*xi / (2 * Unorm)
  return tau_n

tau_T    = tau(T, j_g, k(rho_W, rho_C, rho_A, T, V))

# advective temerature flux :
def L_T_adv(u):
  return c_g * dot(j_g, grad(u))

# advective and diffusive temperature differential operator :
def L_T(u):  
  Lu = + L_T_adv(u) \
       - ( k(rho_W, rho_C, rho_A, u, V)[0] * u.dx(0) ).dx(0) \
       - ( k(rho_W, rho_C, rho_A, u, V)[1] * u.dx(1) ).dx(1)
  return Lu

# enthalpy balance :
T_factor      = (rho_C*c_C + rho_W*c_W + rho_A*c_A + ep*rho_g*c_g)
delta_T       = + T_factor * dudt(T, T1) * beta * dx \
                + L_T_adv(T) * beta * dx \
                + k(rho_W, rho_C, rho_A, T, V)[0] * T.dx(0) * beta.dx(0) * dx \
                + k(rho_W, rho_C, rho_A, T, V)[1] * T.dx(1) * beta.dx(1) * dx \
                - q_r(rho_W, rho_A, T) * beta * dx \
#                + inner(L_T_adv(beta), tau_T*L_T(T)) * dx \
#                - kdTdn(T) * beta * ds \

# total residual :
delta         = delta_j_rho_g + delta_rho_g + delta_rho_W \
                + delta_rho_A + delta_rho_C + delta_T

# Jacobian :
J             = derivative(delta, w, dw)

params      = {'newton_solver' :
                {
                  'linear_solver'           : 'mumps',
                  #'linear_solver'           : 'tfqmr',
                  #'preconditioner'          : 'jacobi',
                  'absolute_tolerance'      : 1e-14,
                  'relative_tolerance'      : 1e-8,
                  'relaxation_parameter'    : 1.0,
                  'maximum_iterations'      : 16,
                  'error_on_nonconvergence' : False
                }
              }
ffc_options = {"optimize"               : True}
#ffc_options = {}

problem = NonlinearVariationalProblem(delta, w, J=J, bcs=bc,
            form_compiler_parameters=ffc_options)
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(params)

# start the timer :
start_time = time()

stars = "*****************************************************************"
initial_dt      = dt
initial_alpha   = params['newton_solver']['relaxation_parameter']
adaptive        = False

# loop over all times :
while t < tf:

  # set the previous solution to the last iteration :
  w1.assign(w)

  # start the timer :
  tic = time()

  # Compute solution
  if not adaptive:
    solver.solve()
  
  ## solve mass equations, lowering time step on failure :
  #if adaptive:
  #  par    = params['newton_solver']
  #  solved_h = False
  #  while not solved_h:
  #    if dt < DOLFIN_EPS:
  #      status_h = [False,False]
  #      break
  #    w_temp   = w.copy(True)
  #    w1_temp  = w1.copy(True)
  #    status_h = solver.solve()
  #    solved_h = status_h[1]
  #    if not solved_h:
  #      dt /= 2.0
  #      print_text(stars, 'red', 1)
  #      s = ">>> WARNING: time step lowered to %g <<<"
  #      print_text(s % dt, 'red', 1)
  #      w.assign(w_temp)
  #      w1.assign(w1_temp)
  #      print_text(stars, 'red', 1)
  
  # solve equation, lower alpha on failure :
  if adaptive:
    solved_u = False
    par    = params['newton_solver']
    while not solved_u:
      if par['relaxation_parameter'] < 0.5:
        status_u = [False, False]
        break
      w_temp   = w.copy(True)
      w1_temp  = w1.copy(True)
      status_u = solver.solve()
      solved_u = status_u[1]
      if not solved_u:
        w.assign(w_temp)
        w1.assign(w1_temp)
        par['relaxation_parameter'] /= 1.4
        print_text(stars, 'red', 1)
        s = ">>> WARNING: newton relaxation parameter lowered to %g <<<"
        print_text(s % par['relaxation_parameter'], 'red', 1)
        print_text(stars, 'red', 1)

  jn, rho_gn,  rho_Wn,  rho_An,  rho_Cn,  Tn = w.split(True)
  
  print_min_max(jn,     'j')
  print_min_max(rho_gn, 'rho_g')
  print_min_max(rho_Wn, 'rho_W')
  print_min_max(rho_An, 'rho_A')
  print_min_max(rho_Cn, 'rho_C')
  print_min_max(Tn,     'T')
  
  # increment time step :
  s = '>>> Time: %g s, CPU time for last dt: %.3f s <<<'
  print_text(s % (t, time()-tic), 'red', 1)

  t += dt
  
  # for the subsequent iteration, reset the parameters to normal :
  if adaptive:
    if par['relaxation_parameter'] != initial_alpha:
      print_text("::: resetting alpha to normal :::", 'green')
      par['relaxation_parameter'] = initial_alpha
    if dt != initial_dt:
      print_text("::: resetting dt to normal :::", 'green')
      dt = initial_dt

# calculate total time to compute
sec = time() - start_time
mnn = sec / 60.0
hor = mnn / 60.0
sec = sec % 60
mnn = mnn % 60
text = "total time to perform transient run: %02d:%02d:%02d" % (hor,mnn,sec)
print_text(text, 'red', 1)

Tn.rename('T', '')
jn.rename('j', '')
rho_gn.rename('rho_g', '')
rho_Wn.rename('rho_W', '')
rho_An.rename('rho_A', '')
rho_Cn.rename('rho_C', '')

File('T.pvd')     << Tn
File('j.pvd')     << jn
File('rho_g.pvd') << rho_gn
File('rho_W.pvd') << rho_Wn
File('rho_A.pvd') << rho_An
File('rho_C.pvd') << rho_Cn


