#!/bin/bash -l
#SBATCH --job-name=abaqus
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --partition=test
#SBATCH --account=project_2008630
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

module load python-data   # fine; container won't use it, but harmless

SIF=/projappl/project_2008630/containers/dolfinx.sif
WORKDIR=$PWD

# One-time install (you can comment these lines out after first success)
rm -rf "$WORKDIR/.pydeps"
mkdir -p "$WORKDIR/.pydeps"
apptainer exec -B "$WORKDIR":"$WORKDIR" -W "$WORKDIR" "$SIF" \
  python -m pip install --no-cache-dir --target "$WORKDIR/.pydeps" "cffi<1.17" "matplotlib<3.9"

# Headless + writable matplotlib cache dir (silences those warnings)
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORKDIR/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# before running:
mkdir -p "$WORKDIR/.cache/fenics" "$WORKDIR/.mplconfig"

# run (no srun shown; add it if you need MPI)
apptainer exec --home "$WORKDIR" -B "$WORKDIR":"$WORKDIR" -W "$WORKDIR" "$SIF" \
  env MPLBACKEND=Agg MPLCONFIGDIR="$WORKDIR/.mplconfig" \
  python "$WORKDIR/test_fenics.py"