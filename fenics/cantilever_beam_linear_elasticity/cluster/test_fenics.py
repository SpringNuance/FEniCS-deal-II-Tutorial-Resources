"""
Solves the equations for linear elasticity in structural mechanics using the
Finite Element Method with FEniCS in Python. The primary unknown is the
displacement u which is a 3D vector field in 3D space.


Cauchy Momentum Equation:           − ∇⋅σ = f

Constitutive Stress-Strain:         σ = λ tr(ε) I₃ + 2 μ ε

Displacement-Strain:                ε = 1/2 (∇u + (∇u)ᵀ)


σ  : Cauchy Stress (3x3 matrix)
f  : Forcing right hand side (3D vector)
λ  : Lambda Lame parameter (scalar)
μ  : Mu Lame parameter (scalar)
ε  : Engineering Strain (3x3 matrix)
I₃ : 3x3 Identity tensor (=matrix)
u  : Displacement (3D vector)

∇⋅ : The divergence operator (here contracts matrix to vector)
tr : The trace operator (sum of elements on main diagonal)
∇  : The gradient operator (here expands vector to matrix)
ᵀ  : The transpose operatore

-------

Scenario:

A cantilever beam is clamped at one end

               .+------------------------+
             .' |                      .'|
            +---+--------------------+'  |      ↓ gravity
   clamped  |   |                    |   |
            |  ,+--------------------+---+
            |.'                      | .'
            +------------------------+'

It is subject to the load due to its own weight and will
deflect accordingly. Under an assumpation of small
deformation the material follows linear elasticity.

------

Solution strategy.:


Define by "v" a test function from the vector function space
on u.

Weak Form:

    <σ(u), ∇v> = <f, v> + <T, v>

with T being the traction vector to prescribe Neumann BC (here =0)


Alternative Weak Form (more commonly used):

    <σ(u), ε(v)> = <f, v> + <T, v>

(valid because σ(u) will always be symmetric and the inner product
of a symmetric matrix with a non-symmetric matrix vanishes)

------

Once the displacement vector field u is obtained, we can compute the
von Mises stress (a scalar stress measure) by

1. Evaluating the deviatoric stress tensor

    s = σ − 1/3 tr(σ) I₃

2. Computing the von Mises stress

    σ_M = √(3/2 s : s)

"""

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".pydeps"))  # <-- use your local deps without touching container PYTHONPATH

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend (safe for clusters)
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx import io

# --- Geometry and discretization parameters
CANTILEVER_LENGTH = 1.0
CANTILEVER_WIDTH = 0.2

N_POINTS_LENGTH = 10
N_POINTS_WIDTH = 3

# --- Material and loading
LAME_MU = 1.0
LAME_LAMBDA = 1.25
DENSITY = 1.0
ACCELERATION_DUE_TO_GRAVITY = 0.016  # downward z

def main():
    comm = MPI.COMM_WORLD

    # ----------------------------
    # Mesh and vector function space
    # ----------------------------
    # Hexahedral box mesh: [0,0,0] x [L,W,W] with (Nx,Ny,Nz) cells
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([CANTILEVER_LENGTH, CANTILEVER_WIDTH, CANTILEVER_WIDTH])],
        [N_POINTS_LENGTH, N_POINTS_WIDTH, N_POINTS_WIDTH],
        cell_type=mesh.CellType.hexahedron,
    )

    # Vector-valued CG1 space using the shape=(gdim,) convention
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

    # ----------------------------
    # Boundary conditions (clamped x=0 face)
    # ----------------------------
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)

    def on_clamped(x):
        return np.isclose(x[0], 0.0)

    facets = mesh.locate_entities_boundary(domain, tdim - 1, on_clamped)
    dofs = fem.locate_dofs_topological(V, tdim - 1, facets)

    zero_vec = fem.Constant(domain, np.array((0.0, 0.0, 0.0), dtype=PETSc.ScalarType))
    bc = fem.dirichletbc(zero_vec, dofs, V)

    # ----------------------------
    # Kinematics and constitutive law
    # ----------------------------
    def epsilon(u):
        # engineering strain ε = sym(grad u)
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        # σ = λ tr(ε) I + 2 μ ε
        return LAME_LAMBDA * ufl.tr(epsilon(u)) * ufl.Identity(domain.geometry.dim) + 2.0 * LAME_MU * epsilon(u)

    # ----------------------------
    # Variational problem
    # ----------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Body force: gravity in -z
    f = fem.Constant(domain, np.array((0.0, 0.0, -DENSITY * ACCELERATION_DUE_TO_GRAVITY), dtype=PETSc.ScalarType))
    traction = fem.Constant(domain, np.array((0.0, 0.0, 0.0), dtype=PETSc.ScalarType))  # zero, included for completeness

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(traction, v) * ufl.ds  # ds is the whole exterior boundary

    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12})
    u_sol = problem.solve()
    u_sol.name = "Displacement Vector"

    # ----------------------------
    # von Mises post-processing
    # ----------------------------
    sig = sigma(u_sol)
    dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(domain.geometry.dim)
    von_mises_expr = ufl.sqrt(1.5 * ufl.inner(dev, dev))  # √(3/2 s:s)

    # Project to CG1 scalar space: find vm ∈ Q s.t. (vm, w) = (expr, w) ∀ w
    Q = fem.functionspace(domain, ("Lagrange", 1))
    vm = fem.Function(Q, name="von Mises stress")
    w = ufl.TestFunction(Q)
    z = ufl.TrialFunction(Q)
    a_proj = ufl.inner(z, w) * ufl.dx
    L_proj = ufl.inner(von_mises_expr, w) * ufl.dx
    proj_problem = LinearProblem(a_proj, L_proj, petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12})
    vm_sol = proj_problem.solve()
    vm.x.array[:] = vm_sol.x.array  # ensure named Function holds result

    # ----------------------------
    # Output (XDMF)
    # ----------------------------
    with io.XDMFFile(comm, "beam_deflection.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_sol, 0.0)
        xdmf.write_function(vm, 0.0)

if __name__ == "__main__":
    main()
