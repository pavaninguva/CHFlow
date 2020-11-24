from dolfin import *

# Simulation constants
theta = 0.01
dt = 0.05
theta_ch = 0.5

# Class for interfacing with the Newton solver
class NavierStokesEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        assemble(self.L, tensor=b)

    def J(self, A, x):
        assemble(self.a, tensor=A)

# Define a simple mesh
N = 10
mesh = UnitSquareMesh(N,N)

# Define Taylor-Hood Elements
# First order finite element for pressure
F1 = FiniteElement("P", mesh.ufl_cell(), 1)
# Second order finite element for velocity
F2 = VectorElement("P", mesh.ufl_cell(), 2)

#Mixed finite element and function space
ME = MixedElement([F2,F1])
NS = FunctionSpace(mesh,ME)

#Create test functions
psi_u, psi_p = TestFunctions(NS)

w = Function(NS)
u, p = split(w)

w_old = Function(NS)
u_old, p_old = split(w_old)

# Define variational forms
# Momentum term
# theta_ch is for fractional timestepping
# theta is our dimensionless viscosity
F_mom = (Constant(1/dt)*dot(u-u_old, psi_u)
        # Viscous term
        + Constant(theta_ch)*Constant(2.0)*inner(theta*sym(grad(u)), sym(grad(psi_u)))
        + Constant(1.0 - theta_ch)*Constant(2.0)*inner(theta*sym(grad(u)), sym(grad(psi_u)))
        # Advective Term
        + Constant(theta_ch)*dot(dot(grad(u),u), psi_u)
        + Constant(1.0-theta_ch)*dot(dot(grad(u),u), psi_u)
        # Pressure Term
        -p*div(psi_u))

F_mass = -psi_p*div(u)

F = (F_mom + F_mass)*dx

# Compute Jacobian
J = derivative(F,w)

# Boundary Conditions
# Apply all Dirichlet BCs. u_x = 1 at the top and zero elsewhere

#Define boundaries
def right(x, on_boundary):
    return x[0] > (1.0 - DOLFIN_EPS)

def left(x,on_boundary):
    return x[0] < DOLFIN_EPS

def top(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS

def bottom(x, on_boundary):
    return x[1] < DOLFIN_EPS

lid_velocity = (1.0, 0.0)

fixed_wall_velocity = (0.0,0.0)


boundary_conditions = [
    DirichletBC(NS.sub(0), lid_velocity, top),
    DirichletBC(NS.sub(0), fixed_wall_velocity, bottom),
    DirichletBC(NS.sub(0), fixed_wall_velocity, left),
    DirichletBC(NS.sub(0), fixed_wall_velocity, right)
]

#Build solver
# problem = NavierStokesEquation(J,F)
# # problem = NonLinearVariationalProblem(F,w,boundary_conditions, J)
# solver = NewtonSolver()
# solver.parameters["linear_solver"] = "lu"
# solver.parameters["convergence_criterion"] = "incremental"
# solver.parameters["relative_tolerance"] = 1e-6

problem = NonlinearVariationalProblem(F,w,boundary_conditions,J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["linear_solver"] = "lu"

#Define output files
file_a = File("u.pvd", "compressed")
file_b = File("p.pvd", "compressed")

#Output initial conditions
t=0.0
timestep = 0

file_a << (w.split()[0],t)
file_b << (w.split()[1],t)

u,p = w.split()

while t < 2.0:

    #Update timestep and t
    timestep +=1
    t += dt
    #Update
    w_old.vector()[:] = w.vector()
    #Solve
    solver.solve()
    
    #Output every 10th timestep
    if timestep %10 == 0:
        file_a << (w.split()[0],t)
        file_b << (w.split()[1],t)