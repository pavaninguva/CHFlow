from dolfin import *
import random
import numpy as np

#Simulation and material parameters to be specified

# Flory-Huggins interaction parameters
chi_AB = 0.15
chi_AC = 0.15
chi_BC = 0.15
# Diffusion coefficient
D_AB = 1.e-10
D_AC = 1.e-10
D_BC = 1.e-10
# Chain length
N_A = 100
N_B = 100
N_C = 100
# Scaling choices 
N_SCALE_OPTION = "N_A"
D_SCALE_OPTION = "D_AB"
# Concentration parameters
A_RAW = 0.1
B_RAW = 0.1
NOISE_MAGNITUDE = 0.03
# Time and space parameters
TIME_MAX = 200
DT = 0.1
N_CELLS = 30
DOMAIN_LENGTH = 400
theta_ch = 1.0

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

P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = MixedElement([P1,P1,P1,P1,P1])
CH = FunctionSpace(mesh, ME, constrained_domain=PeriodicBoundary())

h_1, h_2, j_1, j_2, j_3 = TestFunctions(CH)

ch = Function(CH)
ch0 = Function(CH)

a, b, N_mu_AB, N_mu_AC, N_mu_BC = split(ch)
a0, b0, N_mu0_AB, N_mu0_AC, N_mu0_BC = split(ch0)

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

kappa_AA = (2.0 / 3.0) * chi_AC
kappa_BB = (2.0 / 3.0) * chi_BC
kappa_AB = (1.0 / 3.0) * (chi_AC +  chi_BC - 1.0 * chi_AB)

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

# transport equations
F_a = (
    a * h_1 
    - a0 * h_1
    + dt * a * b * D_AB_ * dot(grad(N_mu_AB_mid), grad(h_1)) 
    + dt * a * (1.0 - a - b) * D_AC_ * dot(grad(N_mu_AC_mid), grad(h_1)) 
)

F_b = (
    b * h_2 
    - b0 * h_2 
    - dt * a * b * D_AB_ * dot(grad(N_mu_AB_mid), grad(h_2)) 
    + dt * b * (1.0 - a - b) * D_BC_ * dot(grad(N_mu_BC_mid), grad(h_2)) 
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

F = (F_a + F_b + F_N_mu_AB + F_N_mu_AC + F_N_mu_BC)*dx(domain=mesh)

# Compute Jacobian
J = derivative(F,ch)

#Set up problem and solver
problem = NonlinearVariationalProblem(F,ch, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["linear_solver"] = "lu"
solver.parameters["newton_solver"]["relaxation_parameter"] = 0.5

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

    ch0.vector()[:] = ch.vector()
    solver.solve()
    
    if timestep % 100 == 0:
        file_a << (ch.split()[0], t)
        file_b << (ch.split()[1], t)

