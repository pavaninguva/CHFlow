from dolfin import *
import random
import numpy as np

#Simulation and material parameters to be specified

# Flory-Huggins interaction parameters
chi_AB = 0.06
chi_AC = 0.06
chi_BC = 0.06
# Diffusion coefficient
D_AB = 1.e-11
D_AC = 1.e-11
D_BC = 1.e-11
# Chain length
N_A = 100
N_B = 100
N_C = 100
# Scaling choices 
N_SCALE_OPTION = "N_A"
D_SCALE_OPTION = "D_AB"
# Concentration parameters
A_RAW = 0.15
B_RAW = 0.15
NOISE_MAGNITUDE = 0.03
# Time and space parameters
TIME_MAX = 100
DT = 0.005
N_CELLS = 40
DOMAIN_LENGTH = 40.0
theta_ch = 1.0
#Viscosity Ratio and fluid params
etaP_etaS = 1
Pe = 1e-1
Sc = 1e4
beta = 1e8


# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = A_RAW + 2.0 * NOISE_MAGNITUDE * (0.5 - random.random())
        values[1] = B_RAW + 2.0 * NOISE_MAGNITUDE * (0.5 - random.random())
        values[2] = 0.0
        values[3] = 0.0
        values[4] = 0.0

    def value_shape(self):
        return (5,)

#Stuff for boundary conditions
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0))

    # Map RightBoundary to LeftBoundary
    def map(self, x, y):
        y[0] = x[0] - DOMAIN_LENGTH
        y[1] = x[1]


# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and build function space
mesh = RectangleMesh(
    Point(0.0, 0.0), Point(DOMAIN_LENGTH, DOMAIN_LENGTH), N_CELLS, N_CELLS
)

# First order lagrangian for CH variables
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# First order finite element for pressure
F1 = FiniteElement("P", mesh.ufl_cell(), 1)
# Second order finite element for velocity
F2 = VectorElement("P", mesh.ufl_cell(), 2)
CH_ME = MixedElement([P1,P1,P1,P1,P1])
NS_ME = MixedElement([F2,F1])
CH = FunctionSpace(mesh, CH_ME, constrained_domain=PeriodicBoundary())
NS = FunctionSpace(mesh,NS_ME, constrained_domain=PeriodicBoundary())


#Impose Dirichlet BC
def top(x, on_boundary):
    return x[1] > DOMAIN_LENGTH - 20*DOLFIN_EPS

def bottom(x, on_boundary):
    return x[1] < DOLFIN_EPS

lid_velocity = (1.0, 0.0)
fixed_wall_velocity = (0.0,0.0)

bcs = [
    DirichletBC(NS.sub(0), lid_velocity, top),
    DirichletBC(NS.sub(0), fixed_wall_velocity, bottom)
]

h_1, h_2, j_1, j_2, j_3 = TestFunctions(CH)
psi_u,psi_p = TestFunctions(NS)

ch = Function(CH)
ch0 = Function(CH)

w = Function(NS)
w0 = Function(NS)

a, b, N_mu_AB, N_mu_AC, N_mu_BC = split(ch)
a0, b0, N_mu0_AB, N_mu0_AC, N_mu0_BC = split(ch0)

u,p = split(w)
u0,p0 = split(w0)

a = variable(a)
b = variable(b)

# Create intial conditions and interpolate
ch_init = InitialConditions(degree=1)
ch.interpolate(ch_init)
ch0.interpolate(ch_init)

N_scale_options = {"N_A": N_A, "N_B": N_B, "N_C": N_C}
N_SCALE = N_scale_options[N_SCALE_OPTION]

D_scale_options = {"D_AB": D_AB, "D_AC": D_AC, "D_BC": D_BC}
D_SCALE = D_scale_options[D_SCALE_OPTION]

kappa_AA = (1.0 / 3.0) * chi_AB
kappa_BB = (1.0 / 3.0) * chi_BC
kappa_AB = (1.0 / 3.0) * (chi_AC +  chi_BC - 2.0 * chi_AB)

N_kappa_AA = N_SCALE * kappa_AA
N_kappa_BB = N_SCALE * kappa_BB
N_kappa_AB = N_SCALE * kappa_AB

# "ordinary free energy" gradients
N_dgda = N_SCALE * (
    (1.0 / N_A) * (1.0 + ln(a)) + chi_AB * b + chi_AC * (1.0 - a - b)
)
N_dgdb = N_SCALE * (
    (1.0 / N_B) * (1.0 + ln(b)) + chi_AB * a + chi_BC * (1.0 - a - b)
)
N_dgdc = N_SCALE * (
    (1.0 / N_C) * (1.0 + ln(1.0 - a - b)) + chi_AC * a +  chi_BC * b
)

N_mu_AB_mid = (1.0 - theta_ch) * N_mu0_AB + theta_ch * N_mu_AB
N_mu_AC_mid = (1.0 - theta_ch) * N_mu0_AC + theta_ch * N_mu_AC
N_mu_BC_mid = (1.0 - theta_ch) * N_mu0_BC + theta_ch * N_mu_BC

# scale diffusivity
D_AB_ = D_AB / D_SCALE
D_AC_ = D_AC / D_SCALE
D_BC_ = D_BC / D_SCALE

dt = DT

#Define theta (viscosity)
theta = (a+b) + (1-a-b)*etaP_etaS


#Transport Equations
# transport equations
F_a = (
    a * h_1 
    - a0 * h_1
    + dt * a * b * D_AB_ * dot(grad(N_mu_AB_mid), grad(h_1)) 
    + dt * a * (1.0 - a - b) * D_AC_ * dot(grad(N_mu_AC_mid), grad(h_1))
    + dt*N_SCALE*Pe*dot(a*u0,grad(h_1)) 
)

F_b = (
    b * h_2 
    - b0 * h_2 
    - dt * a * b * D_AB_ * dot(grad(N_mu_AB_mid), grad(h_2)) 
    + dt * b * (1.0 - a - b) * D_BC_ * dot(grad(N_mu_BC_mid), grad(h_2))
    + dt * N_SCALE*Pe*dot(b*u0, grad(h_2)) 
)

# chemical potential equations
F_N_mu_AB = (
    N_mu_AB * j_1 
    - N_dgda * j_1 
    + N_dgdb * j_1 
    - (N_kappa_AA - N_kappa_AB) * dot(grad(a), grad(j_1)) 
    + (N_kappa_BB - N_kappa_AB) * dot(grad(b), grad(j_1)) 
)

F_N_mu_AC = (
    N_mu_AC * j_2 
    - N_dgda * j_2 
    + N_dgdc * j_2 
    - (N_kappa_AA * dot(grad(a), grad(j_2))) 
    - (N_kappa_AB * dot(grad(b), grad(j_2))) 
)

F_N_mu_BC = (
    N_mu_BC * j_3
    - N_dgdb * j_3
    + N_dgdc * j_3 
    - N_kappa_BB * dot(grad(b), grad(j_3))
    - N_kappa_AB * dot(grad(a), grad(j_3)) 
)

#Navier Stokes
F_mom = (Constant(1/dt)*dot(u-u0, psi_u)
        # Viscous term
        + Constant(N_SCALE)*Constant(Sc)*Constant(theta_ch)*Constant(2.0)*inner(theta*sym(grad(u)), sym(grad(psi_u)))
        + Constant(N_SCALE)*Constant(Sc)*Constant(1.0 - theta_ch)*Constant(2.0)*inner(theta*sym(grad(u)), sym(grad(psi_u)))
        # Advective Term
        + Constant(N_SCALE)*Constant(Pe)*Constant(theta_ch)*dot(dot(grad(u),u), psi_u)
        + Constant(N_SCALE)*Constant(Pe)*Constant(1.0-theta_ch)*dot(dot(grad(u0),u0), psi_u)
        # Pressure Term
        -p*div(psi_u)
        # Coupling Term
        + Constant(beta)*dot((a*b*grad(N_mu_AB_mid) + b*(1.0-a-b)*grad(N_mu_BC_mid)), psi_u)
        )

F_mass = -psi_p*div(u)

F_CH = (F_a+F_b+F_N_mu_AB+F_N_mu_AC+F_N_mu_BC)*dx(domain=mesh)

F_NS = (F_mom + F_mass)*dx(domain=mesh)

#Compute Jacobian
J_CH = derivative(F_CH,ch)
J_NS = derivative(F_NS, w)

#Set up problem and solver
CH_problem = NonlinearVariationalProblem(F_CH,ch, J=J_CH)
CH_solver = NonlinearVariationalSolver(CH_problem)
CH_solver.parameters["newton_solver"]["linear_solver"] = "lu"
CH_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6

NS_problem = NonlinearVariationalProblem(F_NS, w, bcs, J=J_NS) 
NS_solver = NonlinearVariationalSolver(NS_problem) 
NS_solver.parameters["newton_solver"]["linear_solver"] = "mumps"
# NS_solver.parameters["newton_solver"]["preconditioner"] = "hypre_amg"
NS_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6

# Output file
file_a = File("concentration_A.pvd", "compressed")
file_b = File("concentration_B.pvd", "compressed")

# Step in time
t = 0.0
timestep = 0

# output initial condition
file_a << (ch.split()[0], t)
file_b << (ch.split()[1], t)

while t < TIME_MAX:

    timestep += 1
    t += dt

    u0,p0 = split(w0)
    
    ch0.vector()[:] = ch.vector()
    CH_solver.solve()

    w0.vector()[:] = w.vector()
    NS_solver.solve()
    
    if timestep % 50 == 0:
        file_a << (ch.split()[0], t)
        file_b << (ch.split()[1], t)

