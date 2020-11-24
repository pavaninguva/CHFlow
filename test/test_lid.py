from dolfin import *

import matplotlib

N = 4

mesh = UnitSquareMesh(N, N)


P2 = VectorElement('P', mesh.ufl_cell(), 2)

P1 = FiniteElement('P', mesh.ufl_cell(), 1)

P2P1 = MixedElement([P2, P1])

W = FunctionSpace(mesh, P2P1)


psi_u, psi_p = TestFunctions(W)

w = Function(W)

u, p = split(w)


dynamic_viscosity = 0.01

mu = Constant(dynamic_viscosity)


momentum = dot(psi_u, dot(grad(u), u)) - div(psi_u)*p     + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))

mass = -psi_p*div(u)
        
F = (momentum + mass)*dx


JF = derivative(F, w, TrialFunction(W))


lid_velocity = (1., 0.)

lid_location = "near(x[1],  1.)"

fixed_wall_velocity = (0., 0.)

fixed_wall_locations = "near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)"


V = W.sub(0)

boundary_conditions = [
    DirichletBC(V, lid_velocity, lid_location),
    DirichletBC(V, fixed_wall_velocity, fixed_wall_locations)]

problem = NonlinearVariationalProblem(F, w, boundary_conditions, JF)


M = u[0]**2*dx

epsilon_M = 1.e-4

file_a = File("test.pvd")

solver = AdaptiveNonlinearVariationalSolver(problem, M)

solver.solve(epsilon_M)



