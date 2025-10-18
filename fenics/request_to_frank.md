Subject: Request to install FEniCSx (DOLFINx) on the cluster + quick verification test

Hi Frank,

Could you please install FEniCSx (DOLFINx) on the cluster for our FEM/PDE workloads?

According to the official docs, FEniCSx supports three admin-friendly install paths on Linux/HPC:

Spack (recommended for HPC)
The DOLFINx manual explicitly recommends Spack for high-performance computers. 
FEniCS Project

Package: fenics-dolfinx (with Python interface, PETSc/SLEPc, MPI).

After installation, exposing it as a module (e.g., module load fenicsx) would be ideal.

Conda (cluster module or site env)
DOLFINx provides current conda-forge binaries (fenics-dolfinx) that include PETSc/MPI stacks; this is commonly used on clusters and easy to maintain as a module. 
Anaconda
+1

Container (Apptainer/Singularity) from the official Docker image
The FEniCS project maintains official Docker images (dolfinx/dolfinx); we can mirror one to an Apptainer/Singularity image (.sif) and publish it as a module. 
Docker Hub
+1

I’m happy with whichever path aligns best with our cluster standards. If it helps, here are concise commands for each route:

Option A – Spack (HPC-preferred)

Install & load:

spack install fenics-dolfinx ^python@3.11
spack load fenics-dolfinx


(This follows the project’s HPC guidance.) 
FEniCS Project

Option B – Conda (module-backed env)

Create a site env (example):

conda create -n fenicsx -c conda-forge python=3.11 fenics-dolfinx mpich petsc4py slepc4py


(Official conda-forge package reference.) 
Anaconda

(GitHub README also documents conda usage and MPI notes.) 
GitHub

Option C – Apptainer/Singularity (from official Docker)

Mirror the official image:

apptainer pull /shared/images/dolfinx.sif docker://dolfinx/dolfinx:stable


(Official image location.) 
Docker Hub

Quick verification (what success looks like)

After installing, could you please run (or let me run) this tiny Poisson test (based on the official tutorial) to confirm PETSc/MPI and dolfinx are wired correctly? 
FEniCS Project
+1

test_dolfinx_poisson.py

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import mesh, fem, nls, log
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

comm = MPI.COMM_WORLD
rank = comm.rank

# Report versions to verify linkage
if rank == 0:
    print("MPI size:", comm.size)
    print("PETSc version:", PETSc.Sys.getVersion())

# Unit square mesh, CG1 Poisson: -Δu = 1 with u=0 on ∂Ω
domain = mesh.create_unit_square(comm, 32, 32, mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, PETSc.ScalarType(1.0))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Dirichlet BC: u = 0 on the whole boundary
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim-1, tdim)
facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, tdim-1, facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
uh = problem.solve()

# Basic check: L2 norm should be positive and finite
norm_L2 = domain.comm.allreduce(np.sqrt(fem.assemble_scalar(fem.form(uh**2 * ufl.dx))), op=MPI.SUM)
if rank == 0:
    print("Solution L2 norm:", float(norm_L2))


Parallel check (SLURM example) – this just confirms MPI runs through PETSc/dolfinx:

#!/bin/bash
#SBATCH -J fenicsx_check
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:05:00
#SBATCH --mem=2G

# Load whichever module you expose; examples:
# module load fenicsx
# or: module load Miniconda3 && conda activate fenicsx
# or: module load apptainer

# Spack/Conda example:
srun -n ${SLURM_NTASKS} python test_dolfinx_poisson.py

# Apptainer example:
# srun -n ${SLURM_NTASKS} apptainer exec /shared/images/dolfinx.sif \
#     python test_dolfinx_poisson.py


Expected output (master rank):

A line like MPI size: 4

PETSc version: X.Y.Z

Solution L2 norm: <positive number> (value depends slightly on build)

For reference, the official install pages/tutorials I followed are here:

DOLFINx install guide (says Spack is recommended for HPC). 
FEniCS Project

conda-forge fenics-dolfinx package. 
Anaconda

Official Docker images (dolfinx/dolfinx). 
Docker Hub
+1

Official Poisson demo/tutorial to validate the environment. 
FEniCS Project
+1

If you prefer a different packaging method, I’m flexible—whatever best fits our site policy. Thank you for your help.

Best regards,
Nuance