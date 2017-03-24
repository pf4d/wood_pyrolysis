from pylab    import *
from fenics   import *
from time     import time
from helper   import print_text, print_min_max
import ufl

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

parameters['form_compiler']['quadrature_degree'] = 2

# material parameters :
R         = 8.3144      # universal gas constant
T_w       = 273.15      # triple point of water
T_boil    = T_w + 100.0 # boiling point of water
R_v       = 461.5       # water vapour gas constant
R_a       = 287.0       # air gas constant
D_T0      = 7e-6        # diffusion coef. in tangential direction
K         = 5e-16       # specific permeability of dry wood
K_g       = 1.0         # relative permeability of gaseous mixture
zeta      = 0.03        # direction dependent reduction factor
C_0       = 1500.0      # specific heat of dry timber
C_b       = 4200.0      # specific heat of water
C_v       = 1800.0      # specific heat of water vapour
C_a       = 1000.0      # specific heat of air
DH_s      = 2500.0      # heat of sorption
sigma     = 5.67e-8     # Boltzman's constant
eps_g     = 0.6         # porosity  (Turner et al. 2010)

# boundary conditions :
eps_m     = 0.8         # surface emissivity of timber
eps_f     = 1.0         # fire emissivity
alpha_c   = 25.0        # covective heat transfer coefficient
beta      = 2e-7        # mass transfer coefficient
P_g_inf   = 1e5         # ambient gas pressure
rho_v_max = 0.009       # maximum ambient water vapour concentration

# initial conditions :
rho_0     = 445.0       # initial timber density
c_b1      = 53.4        # initial bound water concentration
rho_v1    = 0.009       # initial water vapor concentration
P_g1      = 1e5         # initial gas pressure
T1        = T_w + 20.0  # initial temperature

# mesh parameters :
L         = 0.1         # length
N         = 200         # spatial discretizations
order     = 2           # order of function space

# time parameters :
dt        = 0.01         # time step
t0        = 0.0         # start time
t         = t0          # current time
tf        = 60.0 * 60.0 # final time
eta       = 1           # time-step parameter

# sorption parameters :
b_10d     = 16.3
b_11d     = -0.0367
b_20d     = 2.13
b_21d     = 0.0535
C_1       = 2.7e-4
C_21      = 2.74e-5
C_22      = 19.0
C_3       = 60.0
C_4       = 1e-7

# function spaces : 
mesh = IntervalMesh(N, 0, L)
Q    = FunctionSpace(mesh, 'CG', order)
Q1   = FunctionSpace(mesh, 'CG', 1)
Qe   = FiniteElement('CG', mesh.ufl_cell(), order)
Be   = FiniteElement('B',  mesh.ufl_cell(), 2)
Q4   = FunctionSpace(mesh, MixedElement([Qe]*4))

U    = Function(Q4, name='U')
U0   = Function(Q4, name='U0')
dU   = TrialFunction(Q4)
Phi  = TestFunction(Q4)

Tp     = Function(Q, name = 'T')
P_gp   = Function(Q, name = 'P_g')
rho_vp = Function(Q, name = 'rho_v')
c_bp   = Function(Q, name = 'c_b')

assT     = FunctionAssigner(Q, Q4.sub(0))
assP_g   = FunctionAssigner(Q, Q4.sub(1))
assrho_v = FunctionAssigner(Q, Q4.sub(2))
assc_b   = FunctionAssigner(Q, Q4.sub(3))

# outward-pointing-normal vector :
n    = FacetNormal(mesh)

# cell diameter :
h    = CellSize(mesh)

# cellulosic fire curve (ISO-834) applied ambient temperature :
class AmbientTemperature(Expression):
  def __init__(self, t, element=None):
    self.t = t
  def eval(self, value, x):
    value[0] = T1 + 345*log10(8*self.t/60.0 + 1)
T_inf = AmbientTemperature(t0, element=Qe)

# vapour concentration in ambient air :
class AmbientVapour(Expression):
  def __init__(self, t, element=None):
    self.t = t
  def eval(self, value, x):
    if self.t < 60:
      value[0] = rho_v_max - rho_v_max / 60.0 * self.t
    else:
      value[0] = 0.0
rho_v_inf = AmbientVapour(t0, element=Qe)

class Left(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and abs(x[0]) < 1e-14

class Right(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and abs(x[0] - L) < 1e-14

left  = Left()
right = Right()
ff    = FacetFunction('uint', mesh)
ds    = ds(subdomain_data=ff)

left.mark(ff, 1)
right.mark(ff, 2)

# gas pressure is only essential boundary :
P_g_bc   = DirichletBC(Q4.sub(1), P_g_inf, left)
c_b_bc   = DirichletBC(Q4.sub(3), 0.0,     left)
rho_v_bc = DirichletBC(Q4.sub(2), 0.0,     left)
bcs      = [P_g_bc, c_b_bc]

# temperature, gas pressure, concentration of water vapor, and 
# and concentration of bound water :
T,   P_g,   rho_v,  c_b     = U
T0,  P_g0,  rho_v0, c_b0    = U0
dT,  dP_g,  drho_v, dc_b    = dU
phi, psi,   xi,     kappa   = Phi

# midpoint values :
T_mid     = eta*T     + (1 - eta)*T0
c_b_mid   = eta*c_b   + (1 - eta)*c_b0
rho_v_mid = eta*rho_v + (1 - eta)*rho_v0
P_g_mid   = eta*P_g   + (1 - eta)*P_g0

# set initial conditions:
Ti     = interpolate(Constant(T1),     Q)
P_gi   = interpolate(Constant(P_g1),   Q)
rho_vi = interpolate(Constant(rho_v1), Q)
c_bi   = interpolate(Constant(c_b1),   Q)

# assign initial values :
assign(U,  [Ti, P_gi, rho_vi, c_bi])
assign(U0, [Ti, P_gi, rho_vi, c_bi])

# moisture content :
def m(c_b):
  return c_b / rho_0

# thermal conductivity (Turner 2010) :
def k(c_b):
  return 0.14 + 0.3 * m(c_b)

# bound water activation energy :
def E_b(c_b):
  return (38.5 - 29*m(c_b)) * 1e3

# bound water diffusion:
def D_b(T, c_b):
  return D_T0 * exp(- E_b(c_b) / (R*T))

# bound water temperature diffusion :
def D_bT(T, c_b):
  return D_b(T, c_b) * c_b * E_b(c_b) / (R*T**2)

# viscosity of gas mixture (Gronli pp135) :
def mu_g(T):
  return 7.85e-6 + 2.18e-8 * T

# gas velocity :
def v_g(T, P_g):
  return Constant(0.0)#K * K_g / mu_g(T) * P_g.dx(0)

# diffusion coefficient of air -> water vapor :
def D_av(T, P_g):
  return zeta * 1.87 * T**2.072 / P_g * 1e-5

# partial vapor pressure :
def P_v(T, rho_v):
  return R_v * rho_v * T

# partial air pressure :
def P_a(T, rho_v, P_g):
  return P_g - P_v(T, rho_v)

# concentration of air vapour :
def rho_a(T, P_g, rho_v):
  return P_a(T, rho_v, P_g) / (R_a * T)

# gas mixture concentration :
def rho_g(T, P_g, rho_v):
  return rho_v + rho_a(T, P_g, rho_v)

# saturated vapor pressure (Gronli p119) :
def P_s(T):
  return exp(24.1201 - 4671.3545 / T)

# relative humidity :
def h_m(T, rho_v):
  return P_v(T, rho_v) / P_s(T)

# equilibrium moisture content :
def c_bl(T, rho_v):
  f_1   = b_10d + b_11d*T
  f_2   = b_20d + b_21d*T
  return ln( ln(1/h_m(T, rho_v))/f_1 ) / f_2 * rho_0
  
  # reaction rate function :
def H_c(T, rho_v, c_b):
  C_2   = C_21 * exp(C_22*h_m(T, rho_v))   # another coefficent
  H_c_n = conditional(le(c_b, c_bl(T, rho_v)), \
                      C_1*exp(-C_2*(    c_b/c_bl(T, rho_v))**C_3) + C_4, \
                      C_1*exp(-C_2*(2 - c_b/c_bl(T, rho_v))**C_3) + C_4 )
  return H_c_n

# sorption rate :
def cdot(T, rho_v, c_b):
  cdot_n = conditional( le(T, T_boil), \
                        H_c(T, rho_v, c_b)*(c_bl(T, rho_v) - c_b), \
                        H_c(T, rho_v, c_b)*(0 - c_b) )
  cdot_n = Constant(0.0)
  return cdot_n

# thermal conductivity :
def k(c_b):
  return 0.14 + 0.3 * m(c_b)

# flux of bound water :
def J_b(T, c_b):
  return - D_b(T, c_b)*c_b.dx(0) - D_bT(T, c_b)*T.dx(0)

# flux of air :
def J_a(T, P_g, rho_v):
  J_a_n = + eps_g * rho_a(T, P_g, rho_v) * v_g(T, P_g) \
          - eps_g * rho_g(T, P_g, rho_v) \
          * D_av(T, P_g) * (rho_a(T, P_g, rho_v) / rho_g(T, P_g, rho_v)).dx(0)
  return J_a_n

# flux of vapor :
def J_v(T, P_g, rho_v):
  J_v_n = + eps_g * rho_v * v_g(T, P_g) \
          - eps_g * rho_g(T, P_g, rho_v) \
          * D_av(T, P_g) * (rho_v / rho_g(T, P_g, rho_v)).dx(0)
  return J_v_n

# time derivative :
def dudt(u,u0):  return (u - u0) / dt

# energy residual :
conv        = + C_b*J_b(T_mid, c_b_mid) \
              + C_v*J_v(T_mid, P_g_mid, rho_v_mid) \
              + C_a*J_a(T_mid, P_g_mid, rho_v_mid) \
              - k(c_b_mid).dx(0) + 1e-10
Pe_T        = h * conv / (2*k(c_b_mid))
tau_T       = h / (2*conv) * (1/tanh(Pe_T) - 1 /Pe_T)

# discrete time derivative of sorption rate :
dcdotdt   = ( cdot(T, rho_v, c_b) - cdot(T0, rho_v0, c_b0) ) / dt

# midpoint values of sorption rate :
cdot_mid  = eta*cdot(T, rho_v, c_b) + (1 - eta)*cdot(T0, rho_v0, c_b0)

def L_T_adv(u):
  Lu = ( + C_b*J_b(u, c_b_mid) \
         + C_v*J_v(u, P_g_mid, rho_v_mid) \
         + C_a*J_a(u, P_g_mid, rho_v_mid) - k(c_b_mid).dx(0)) * u.dx(0)
  return Lu

def L_T(u):  
  Lu = + (k(c_b_mid) * u.dx(0)).dx(0) \
       - DH_s * cdot(u, rho_v_mid, c_b_mid) \
       - ( + C_b*J_b(u, c_b_mid) \
           + C_v*J_v(u, P_g_mid, rho_v_mid) \
           + C_a*J_a(u, P_g_mid, rho_v_mid) ) * u.dx(0)
  return Lu

dTdt        = dudt(T, T0)
h_c         = alpha_c * (T_inf - T_mid)
h_r         = sigma * eps_m * eps_f * (T_inf**4 - T_mid**4)
kdTdn       = h_c + h_r
delta_T     = + ( + eps_g * rho_a(T_mid, P_g_mid, rho_v_mid) * C_a \
                  + eps_g * rho_v_mid * C_v \
                  + c_b_mid * C_b \
                  + rho_0 * C_0 ) * dTdt * phi * dx \
              + DH_s * cdot_mid * phi * dx \
              + conv * T_mid.dx(0) * phi * dx \
              + k(c_b_mid) * T_mid.dx(0) * phi.dx(0) * dx \
              - kdTdn * phi * ds(1) \
#              + inner(L_T_adv(phi), tau_T*L_T(T_mid)) * dx

# bound water residual :
dc_bdt      = dudt(c_b, c_b0)
kappa_c     = D_b(T_mid, c_b_mid) + 1e-10
d_c         = D_b(T_mid, c_b_mid).dx(0) + 1e-10
s_c         = - D_b(T_mid, c_b0) * E_b(c_b0) \
              * (T_mid.dx(0).dx(0)) / (R*T_mid**2)
Pe_c        = h * d_c / (2*kappa_c)
tau_c       = 1 / (4*kappa_c/h**2 + 2*d_c/h + s_c)
def L_c(u):      return J_b(T_mid, u).dx(0)
def L_c_adv(u):  return u * d_c.dx(0) + d_c * u.dx(0)
def R_c(u,u0):   return dudt(u, u0) + L_c(u) - cdot(T_mid, rho_v_mid, u)
delta_c_b   = + dc_bdt * psi * dx \
              - cdot_mid * psi * dx \
              - J_b(T_mid, c_b_mid) * psi.dx(0) * dx \
#              + inner(L_c_adv(psi), tau_c*R_c(c_b, c_b0)) * dx \

# water vapor residual :
drho_vdt    = dudt(rho_v, rho_v0)
kappa_rho   = eps_g * rho_g(T, P_g, rho_v) * D_av(T_mid, P_g_mid)
d_rho       = eps_g * v_g(T_mid, P_g_mid) + 1e-10
Pe_rho      = h * d_rho / (2*kappa_rho)
tau_rho     = h / (2*d_rho) * (1/tanh(Pe_rho) - 1 / Pe_rho)
def L_rho_v(u):     return J_v(T_mid, P_g_mid, u).dx(0)
def L_rho_adv(u):   return u * d_rho.dx(0) + d_rho * u.dx(0)
def R_rho_v(u,u0):  return dudt(u, u0) + L_rho_v(u) - cdot(T_mid, u, c_b_mid)
J_vdn       = - beta * (rho_v_inf - rho_v_mid)
delta_rho_v = + eps_g * drho_vdt * xi * dx \
              - J_v(T_mid, P_g_mid, rho_v_mid) * xi.dx(0) * dx \
              + J_vdn * xi * ds(1) \
              + cdot_mid * xi * dx \
#              + inner(L_rho_adv(xi), tau_rho*R_rho_v(rho_v, rho_v0)) * dx

# gas pressure residual :
def drho_adt(P, P0):
  drho_adt_n  = + 1 / (R_a*T_mid) * dudt(P, P0) \
                - P / (R_a*T_mid**2) * dTdt \
                - R_v / R_a * drho_vdt
  return drho_adt_n

kappa_P     = eps_g * rho_g(T, P_g, rho_v) * D_av(T_mid, P_g_mid)
d_P         = eps_g * v_g(T_mid, P_g_mid) + 1e-10
Pe_P        = h * d_P / (2*kappa_P)
tau_P       = h / (2*d_P) * (1/tanh(Pe_P) - 1 / Pe_P)
def L_P_adv(u):   return eps_g * ( + u * v_g(T_mid, u).dx(0) \
                                   + v_g(T_mid, u) * u.dx(0) )
def L_P(u):       return J_a(T_mid, u, rho_v_mid).dx(0)
def R_P(u,u0):    return drho_adt(u, u0) + L_P(u)
delta_P_g   = + eps_g * drho_adt(P_g, P_g0) * kappa * dx \
              - J_a(T_mid, P_g_mid, rho_v_mid) * kappa.dx(0) * dx \
              + J_a(T_mid, P_g_mid, rho_v_mid) * kappa * ds(2) \
#              + inner(L_P_adv(kappa), tau_rho*R_P(P_g, P_g0)) * dx

# mixed formulation :
delta       = delta_T + delta_c_b + delta_rho_v + delta_P_g

# Jacobian :
J           = derivative(delta, U, dU)

params      = {'newton_solver' :
                {
                  'linear_solver'           : 'mumps',
                  #'preconditioner'          : 'hypre_amg',
                  'absolute_tolerance'      : 1e-8,
                  'relative_tolerance'      : 1e-5,
                  'relaxation_parameter'    : 1.0,
                  'maximum_iterations'      : 10,
                  'error_on_nonconvergence' : False
                }
              }
ffc_options = {"optimize"               : True,
               "eliminate_zeros"        : True,
               "precompute_basis_const" : True,
               "precompute_ip_const"    : True}

problem = NonlinearVariationalProblem(delta, U, J=J, bcs=bcs,
            form_compiler_parameters=ffc_options)
solver = NonlinearVariationalSolver(problem)
solver.parameters.update(params)


#===============================================================================
# set up visualization :

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage{fouriernc}']
 
  
def calculate_plot_variables(T, P_g, rho_v, c_b):
  
  v_gp = project(v_g(T, P_g), Q1)

  if order != 1:
    T     = interpolate(T, Q1)
    P_g   = interpolate(P_g, Q1)
    rho_v = interpolate(rho_v, Q1)
    c_b   = interpolate(c_b, Q1)
    v_gp  = interpolate(v_gp, Q1)

  T_v     = T.vector().array()[::-1]
  P_g_v   = P_g.vector().array()[::-1]
  rho_v_v = rho_v.vector().array()[::-1]
  c_b_v   = c_b.vector().array()[::-1]
  v_g_v   = v_gp.vector().array()[::-1]
  
  Tf      = T_v - T_w
  mf      = 100 * m(c_b_v)
  P_gf    = P_g_v / 1e6
  rho_af  = rho_a(T_v, P_g_v, rho_v_v) * 1e3
  rho_vf  = rho_v_v * 1e3
  v_gf    = v_g_v

  return Tf, mf, rho_af, rho_vf, P_gf, v_gf


x = 100 * mesh.coordinates()[:,0]

plt.ion()

fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.set_xlim(0.0, L*100)
ax2.set_xlim(0.0, L*100)
ax3.set_xlim(0.0, L*100)
ax4.set_xlim(0.0, L*100)
ax6.set_xlim(0.0, L*100)

#ax1.set_ylim(0.0, 40.0)
#ax2.set_ylim(0.0, 14.0)
#ax3.set_ylim(0.1, 0.18)
#ax4.set_ylim(0.0, 700.0)

out = calculate_plot_variables(Tp, P_gp, rho_vp, c_bp)
Tf, mf, rho_af, rho_vf, P_gf, v_gf = out

Tplt,  = ax1.plot(x, Tf,     'k',   lw=2.0, label=r"$T$")
mplt,  = ax2.plot(x, mf,     'k',   lw=2.0, label=r"$m$")
Pplt,  = ax3.plot(x, P_gf,   'k',   lw=2.0, label=r"$P_g$")
raplt, = ax4.plot(x, rho_af, 'k',   lw=2.0, label=r"$\rho_a$")
rvplt, = ax5.plot(x, rho_vf, 'k',   lw=2.0, label=r"$\rho_v$")
vplt,  = ax6.plot(x, v_gf,   'k',   lw=2.0, label=r"$v_g$")

#leg = ax.legend(loc='upper left', ncol=2, fontsize='medium')
#leg.get_frame().set_alpha(0.0)


ax1.set_title(r'Temperature $T$ [$^{\circ}$C]')
ax1.set_xlabel(r'$x$ [cm]')
#ax1.set_ylabel(r'$T$ [$^{\circ}$C]')
ax1.grid()

ax2.set_title(r'Moisture content $m$ [\%]')
ax2.set_xlabel(r'$x$ [cm]')
#ax2.set_ylabel(r'$m$ [\%]')
ax2.grid()

ax3.set_title(r'Gas pressure $P_g$ [MPa]')
ax3.set_xlabel(r'$x$ [cm]')
#ax3.set_ylabel(r'$P_g$ [MPa]')
ax3.grid()

ax4.set_title(r'Air concentration $\rho_a$ [g/m$^3$]')
ax4.set_xlabel(r'$x$ [cm]')
#ax4.set_ylabel(r'$\rho_a$ [g/m$^3$]')
ax4.grid()

ax5.set_title(r'Vapor concentration $\rho_v$ [g/m$^3$]')
ax5.set_xlabel(r'$x$ [cm]')
#ax5.set_ylabel(r'$\rho_v$ [g/m$^3$]')
ax5.grid()

ax6.set_title(r'Gas velocity $v_g$ [m/s]')
ax6.set_xlabel(r'$x$ [cm]')
#ax6.set_ylabel(r'$v_g$ [m/s]')
ax6.grid()

plt.tight_layout()
plt.show()


def update_plot(T, P_g, rho_v, c_b):
 
  out = calculate_plot_variables(T, P_g, rho_v, c_b)
  Tf, mf, rho_af, rho_vf, P_gf, v_gf = out
  
  ax1.set_ylim(Tf.min(),     Tf.max())
  ax2.set_ylim(mf.min(),     mf.max())
  ax3.set_ylim(P_gf.min(),   P_gf.max())
  ax4.set_ylim(rho_af.min(), rho_af.max())
  ax5.set_ylim(rho_vf.min(), rho_vf.max())
  ax6.set_ylim(v_gf.min(),   v_gf.max())
  
  Tplt.set_ydata(Tf)
  mplt.set_ydata(mf)
  Pplt.set_ydata(P_gf)
  raplt.set_ydata(rho_af)
  rvplt.set_ydata(rho_vf)
  vplt.set_ydata(v_gf)

  plt.draw()
  plt.pause(0.00000001)


def solve_and_plot():

  # solve nonlinear system :
  rtol   = params['newton_solver']['relative_tolerance']
  maxit  = params['newton_solver']['maximum_iterations']
  alpha  = params['newton_solver']['relaxation_parameter']
  s      = "::: solving problem with %i max iterations" + \
           " and step size = %.1f :::"
  print_text(s % (maxit, alpha), 'dark_orange_3a')
  
  # compute solution :
  #solve(delta == 0, U, bcs, J=J, solver_parameters=params)
  out = solver.solve()

  # set the previous solution :
  U0.assign(U)

  assT.assign(Tp,         U.sub(0))
  assP_g.assign(P_gp,     U.sub(1))
  assrho_v.assign(rho_vp, U.sub(2))
  assc_b.assign(c_bp,     U.sub(3))

  print_min_max(Tp,     'T')
  print_min_max(P_gp,   'P_g')
  print_min_max(rho_vp, 'rho_v')
  print_min_max(c_bp,   'c_b')
  
  update_plot(Tp, P_gp, rho_vp, c_bp)

  return out


stars = "*****************************************************************"
step_time       = []
initial_dt      = dt
initial_alpha   = params['newton_solver']['relaxation_parameter']
adaptive        = True

# start the timer :
start_time = time()

# Loop over all times
while t < tf:

  # start the timer :
  tic = time()
  
  ## solve equation, lower alpha on failure :
  #if adaptive:
  #  solved_u = False
  #  par    = params['newton_solver']
  #  while not solved_u:
  #    if par['relaxation_parameter'] < 0.2:
  #      status_u = [False, False]
  #      break
  #    status_u = solve_and_plot()
  #    solved_u = status_u[1]
  #    if not solved_u:
  #      par['relaxation_parameter'] /= 1.43
  #      print_text(stars, 'red', 1)
  #      s = ">>> WARNING: newton relaxation parameter lowered to %g <<<"
  #      print_text(s % par['relaxation_parameter'], 'red', 1)
  #      print_text(stars, 'red', 1)

  # solve mass equations, lowering time step on failure :
  if adaptive:
    par    = params['newton_solver']
    solved_h = False
    while not solved_h:
      if dt < DOLFIN_EPS:
        status_h = [False,False]
        break
      U_temp   = U.copy(True)
      U0_temp  = U0.copy(True)
      status_h = solve_and_plot()
      solved_h = status_h[1]
      if not solved_h:
        dt /= 2.0
        print_text(stars, 'red', 1)
        s = ">>> WARNING: time step lowered to %g <<<"
        print_text(s % dt, 'red', 1)
        U.assign(U_temp)
        U0.assign(U0_temp)
        print_text(stars, 'red', 1)

  # solve :
  else:
    solve_and_plot()

  # increment time step :
  s = '>>> Time: %g s, CPU time for last dt: %.3f s <<<'
  print_text(s % (t+dt, time()-tic), 'red', 1)

  t += dt
  step_time.append(time() - tic)

  # increment time in boundary conditions:
  T_inf.t     = t
  rho_v_inf.t = t
  
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

plt.ioff()
plt.show()



