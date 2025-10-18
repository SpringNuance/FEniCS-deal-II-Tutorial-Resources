# Implicit Euler solution of 1D heat equation on [0,1] with u=0 at x=0,1
# u_t - u_xx = f,  u(0,x)=sin(pi x), f=0
# Exports a PNG with curves at t = 0, dt, 2dt, ...

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".pydeps"))  # <-- use your local deps without touching container PYTHONPATH

# Implicit Euler solution of 1D heat equation on [0,1] with u=0 at x=0,1
# u_t - u_xx = f,  u(0,x)=sin(pi x), f=0
# Exports a PNG with curves at t = 0, dt, 2dt, ...

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend (safe for clusters)
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # --- Discretization: [0,1], CG1
    n_elements = 32
    domain = mesh.create_unit_interval(comm, n_elements)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # --- Homogeneous Dirichlet BCs on x=0 and x=1
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim-1, tdim)

    def on_left(x):  return np.isclose(x[0], 0.0)
    def on_right(x): return np.isclose(x[0], 1.0)

    left_facets  = mesh.locate_entities_boundary(domain, tdim-1, on_left)
    right_facets = mesh.locate_entities_boundary(domain, tdim-1, on_right)

    dofs_L = fem.locate_dofs_topological(V, tdim-1, left_facets)
    dofs_R = fem.locate_dofs_topological(V, tdim-1, right_facets)

    bc0 = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_L, V)
    bc1 = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_R, V)
    bcs = [bc0, bc1]

    # --- Initial condition u(0,x) = sin(pi x)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]))

    # --- Time stepping parameters
    dt = 0.1
    num_steps = 5
    f = fem.Constant(domain, PETSc.ScalarType(0.0))  # heat source

    # --- Variational forms: (u, v) + dt*(grad u, grad v) = (u_n, v) + dt*(f, v)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f * v) * ufl.dx

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
    u_sol = fem.Function(V, name="u")

    # --- Gather coordinates for plotting (rank 0)
    # --- Plotting setup: use OWNED dofs only (coords and values match in length)
    index_map = V.dofmap.index_map
    num_owned = index_map.size_local * V.dofmap.bs  # bs=1 here
    xdofs = V.tabulate_dof_coordinates().reshape((-1, 1))
    xs_owned = xdofs[:num_owned, 0]
    order = np.argsort(xs_owned)


    if rank == 0:
        plt.figure()
        # initial plot
        plt.plot(xs_owned[order], u_n.x.array[order], label="t=0.0")


    # --- Time loop
    t = 0.0
    for _ in range(num_steps):
        t += dt
        u_sol = problem.solve()            # assembles with current u_n
        u_n.x.array[:] = u_sol.x.array     # advance solution

        if rank == 0:
            plt.plot(xs_owned[order], u_sol.x.array[order], label=f"t={t:1.1f}")

    # --- Save figure (rank 0)
    if rank == 0:
        plt.legend()
        plt.title("Heat conduction in a rod (Dirichlet u=0 at x=0,1)")
        plt.xlabel("x position")
        plt.ylabel("Temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("heat_1d.png", dpi=200)

if __name__ == "__main__":
    main()
