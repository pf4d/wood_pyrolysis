from pylab  import *
from fenics import *
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
rho_v1    = 0.009       # initial water vapour concentration
P_g1      = 1e5         # initial gas pressure
T1        = T_w + 20.0  # initial temperature

# mesh parameters :
L         = 0.005       # length
N         = 1000        # spatial discretizations
order     = 2           # order of function space

# time parameters :
dt        = 1.0         # time step
t0        = 0.0         # start time
t         = t0          # current time
tf        = 56.0        # final time

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

# diffusion coefficient parameters :
a         = 1.87
b         = 2.072
c         = 1e-5

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

Tp     = Function(Q)
P_gp   = Function(Q)
rho_vp = Function(Q)
c_bp   = Function(Q)

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

# time-step parameter (Crank-Nicolson) and midpoint values :
eta       = 0.5
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

m     = c_b_mid / rho_0                    # moisture content
k     = 0.14 + 0.3 * m                     # thermal conductivity (Turner 2010)
E_b   = (38.5 - 29*m) * 1e3                # bound water activation energy
D_b   = D_T0 * exp(- E_b / (R*T_mid) )     # bound water diffusion
D_bT  = D_b * c_b_mid*E_b / (R*T_mid**2)   # bound water temperature diffusion
mu_g  = 7.85e-6 + 2.62e-8 * T_mid          # viscosity of gas mixture
v_g   = K * K_g / mu_g * P_g_mid.dx(0)     # gas velocity
D_av  = zeta * a * T_mid**b / P_g_mid * c  # diff. coef. of air -> w. vapour
P_v   = R_v * rho_v_mid * T_mid            # partial vapor pressure
P_a   = P_g_mid - P_v                      # partial air pressure
rho_a = P_a / (R_a * T_mid)                # concentration of air vapour
rho_g = rho_v_mid + rho_a                  # gas mixture concentration
P_s   = exp(24.1201 - 4671.3545 / T_mid)   # saturated vapour pressure
h_m   = P_v / P_s                          # relative humidity
C_2   = C_21 * exp(C_22*h_m)               # another coefficent

# flux of bound water, air, and vapour :
J_b   = - D_b * c_b_mid.dx(0) \
        - D_bT * T_mid.dx(0)
J_a   = + eps_g * rho_a * v_g \
        - eps_g * rho_g * D_av * (rho_a / rho_g).dx(0)
J_v   = + eps_g * rho_v_mid * v_g \
        - eps_g * rho_g * D_av * (rho_v_mid / rho_g).dx(0)

#K_VT  = (D_av * eps_g * P_g_mid * rho_v_mid) / (R_a * T_mid**2 * rho_g)
#K_VP  = eps_g * rho_v_mid * ((K * K_g / mu_g - D_av)/(R_a * T_mid * rho_g))
#K_W   = D_av * eps_g / rho_g * (rho_a + R_v / R_a * rho_v_mid)
#J_v   = - K_VT * T_mid.dx(0) - K_VP * P_g_mid.dx(0) - K_W * rho_v_mid.dx(0)
#
#K_AT  = (D_av * eps_g * P_g_mid * rho_v_mid) / (R_a * T_mid**2 * rho_g)
#K_AP  = + K * K_g / mu_g * eps_g * rho_a \
#        + (D_av * eps_g * rho_v_mid) / (R_a * T_mid * rho_g)
#K_AV  = D_av * eps_g / rho_g * (- rho_a - R_v / R_a * rho_v_mid)
#J_a   = K_AT * T_mid.dx(0) - K_AP * P_g_mid.dx(0) - K_AV * rho_v_mid.dx(0)

# equilibrium moisture content :
f_1   = b_10d + b_11d*T_mid
f_2   = b_20d + b_21d*T_mid
c_bl  = ln( ln(1/h_m)/f_1 ) / f_2 * rho_0

# reaction rate function :
H_c   = conditional(le(c_b_mid, c_bl), \
                    C_1*exp(-C_2*(    c_b_mid/c_bl)**C_3) + C_4, \
                    C_1*exp(-C_2*(2 - c_b_mid/c_bl)**C_3) + C_4 )

# sorption rate :
cdot  = conditional( le(T, T_boil), H_c*(c_bl - c_b_mid), H_c*(0 - c_b_mid) )
#cdot = Constant(0.0)

# boundary conditions :
h_c   = alpha_c * (T_inf - T_mid)
h_r   = sigma * eps_m * eps_f * (T_inf**4 - T_mid**4)
kdTdn = h_c + h_r
J_vdn = - beta * (rho_v_inf - rho_v_mid)

## for SUPG :
##Pe  = h * d / (2*kappa)
##tau = h / (2*d) * (1/tanh(Pe) - 1 / Pe)
#
## for GLS or SSM :
#tau = 1 / (4*kappa/h**2 + 2*d/h + s)
#
#def L(u):       return -(kappa * u.dx(0)).dx(0) + d*u.dx(0) + s*u  # GLS
#def L_star(u):  return -(kappa * u.dx(0)).dx(0) - d*u.dx(0) + s*u  # SSM
#def L_adv(u):   return d*u.dx(0)                                   # SUPG


# energy residual :
dTdt        = (T - T0) / dt
delta_T     = + ( + eps_g * rho_a * C_a \
                  + eps_g * rho_v * C_v \
                  + c_b * C_b \
                  + rho_0 * C_0 ) * dTdt * phi * dx \
              + DH_s * cdot * phi * dx \
              + ( C_b*J_b + C_v*J_v + C_a*J_a ) * T_mid.dx(0) * phi * dx \
              + k * T_mid.dx(0) * phi.dx(0) * dx \
              - kdTdn * phi * ds(1)

# bound water residual :
dc_bdt      = (c_b - c_b0) / dt
delta_c_b   = + dc_bdt * psi * dx \
              - J_b * psi.dx(0) * dx \
              - cdot * psi * dx

# water vapor residual :
drho_vdt    = (rho_v - rho_v0) / dt
delta_rho_v = + eps_g * drho_vdt * xi * dx \
              - J_v * xi.dx(0) * dx \
              + J_vdn * xi * ds(1) \
              + cdot * xi * dx

# gas pressure residual :
dP_gdt      = (P_g - P_g0) / dt
drho_adt    = + 1 / (R_a*T_mid) * dP_gdt \
              - 1 / (R_a*T_mid**2) * dTdt * P_g_mid \
              - R_v / R_a * drho_vdt
delta_P_g   = + eps_g * drho_adt * kappa * dx \
              - J_a * kappa.dx(0) * dx \
              + J_a * kappa * Constant(-1) * ds(1) \
              + J_a * kappa * Constant(1)  * ds(2)

# mixed formulation :
delta       = delta_T + delta_c_b + delta_rho_v + delta_P_g

# Jacobian :
J           = derivative(delta, U, dU)

params      = {'newton_solver' :
                {
                  'linear_solver'           : 'mumps',
                  'relative_tolerance'      : 1e-8,
                  'relaxation_parameter'    : 1.0,
                  'maximum_iterations'      : 3,
                  'error_on_nonconvergence' : False
                }
              }

# set up visualization :
Tf1     = Function(Q1)
P_gf1   = Function(Q1)
rho_vf1 = Function(Q1)
c_bf1   = Function(Q1)

Tf1.interpolate(Ti)
P_gf1.interpolate(P_gi)
rho_vf1.interpolate(rho_vi)
c_bf1.interpolate(c_bi)

Tf      = Tf1.vector().array()[::-1]
P_gf    = P_gf1.vector().array()[::-1]
rho_vf  = rho_vf1.vector().array()[::-1]
c_bf    = c_bf1.vector().array()[::-1]

x       = 100 * mesh.coordinates()[:,0]
Tf      = Tf - T_w
mf      = 100 * c_bf / rho_0
P_gf    = P_gf / 1e6
rho_vf  = rho_vf * 1e3

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage{fouriernc}']

plt.ion()

fig = figure(figsize=(10,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.set_xlim(0.0, L*100)
ax2.set_xlim(0.0, L*100)
ax3.set_xlim(0.0, L*100)
ax4.set_xlim(0.0, L*100)

#ax1.set_ylim(0.0, 40.0)
#ax2.set_ylim(0.0, 14.0)
#ax3.set_ylim(0.1, 0.18)
#ax4.set_ylim(0.0, 700.0)

Tplt, = ax1.plot(x, Tf,     'k',   lw=2.0, label=r"$T$")
mplt, = ax2.plot(x, mf,     'k',   lw=2.0, label=r"$m$")
Pplt, = ax3.plot(x, P_gf,   'k',   lw=2.0, label=r"$P_g$")
rplt, = ax4.plot(x, rho_vf, 'k',   lw=2.0, label=r"$\rho_v$")

#leg = ax.legend(loc='upper left', ncol=2, fontsize='medium')
#leg.get_frame().set_alpha(0.0)


ax1.set_title('Temperature')
ax1.set_xlabel(r'$x$ [cm]')
ax1.set_ylabel(r'$T$ [$^{\circ}$ C]')
ax1.grid()

ax2.set_title('Moisture content')
ax2.set_xlabel(r'$x$ [cm]')
ax2.set_ylabel(r'$m$ [\%]')
ax2.grid()

ax3.set_title('Gas pressure')
ax3.set_xlabel(r'$x$ [cm]')
ax3.set_ylabel(r'$P_g$ [MPa]')
ax3.grid()

ax4.set_title('Vapour concentration')
ax4.set_xlabel(r'$x$ [cm]')
ax4.set_ylabel(r'$\rho_v$ [kg/m^3]')
ax4.grid()

plt.tight_layout()
plt.show()

# time stepping :
while t < tf:

  # solve :
  #params['newton_solver']['relative_tolerance'] = 1e-8
  solve(delta == 0, U, bcs, J=J, solver_parameters=params)

  # set the previous solution :
  U0.assign(U)

  # increment time step :
  t += dt

  # increment time in boundary conditions:
  T_inf.t   = t
  rho_v_inf = t

  assT.assign(Tp,         U.sub(0))
  assP_g.assign(P_gp,     U.sub(1))
  assrho_v.assign(rho_vp, U.sub(2))
  assc_b.assign(c_bp,     U.sub(3))

  print Tp.vector().min(),     Tp.vector().max()
  print P_gp.vector().min(),   P_gp.vector().max()
  print rho_vp.vector().min(), rho_vp.vector().max()
  print c_bp.vector().min(),   c_bp.vector().max()
  print "t:", t

  if order != 1:
    Tf1.interpolate(Tp)
    P_gf1.interpolate(P_gp)
    rho_vf1.interpolate(rho_vp)
    c_bf1.interpolate(c_bp)
  else:
    Tf1     = Tp
    P_gf1   = P_gp
    rho_vf1 = rho_vp
    c_bf1   = c_bp
  
  Tf      = Tf1.vector().array()[::-1]
  P_gf    = P_gf1.vector().array()[::-1]
  rho_vf  = rho_vf1.vector().array()[::-1]
  c_bf    = c_bf1.vector().array()[::-1]
  
  Tf      = Tf - T_w
  mf      = 100 * c_bf / rho_0
  P_gf    = P_gf / 1e6
  rho_vf  = rho_vf * 1e3

  ax1.set_ylim(Tf.min(),     Tf.max())
  ax2.set_ylim(mf.min(),     mf.max())
  ax3.set_ylim(P_gf.min(),   P_gf.max())
  ax4.set_ylim(rho_vf.min(), rho_vf.max())
  
  Tplt.set_ydata(Tf)
  mplt.set_ydata(mf)
  Pplt.set_ydata(P_gf)
  rplt.set_ydata(rho_vf) 
  plt.draw()
  plt.pause(0.00000001)

plt.ioff()
plt.show()



