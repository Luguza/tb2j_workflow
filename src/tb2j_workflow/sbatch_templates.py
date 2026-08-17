SCF_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=bw26b012
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time={time}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir={output_dir}

module load {vasp_module}

vasp
"""

WANNIER90_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=bw26b012
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time={time}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir={output_dir}

set -e

module load {vasp_module}

# -f Wan90 selects the VASP build linked against the wannier90 library. That
# build stops well short of a wannierisation: it calls wannier_setup, writes
# the overlap and projection matrices (wannier90.{{1,2}}.{{amn,mmn,eig}}) and does
# a one-shot SVD, then ends. It never calls wannier_run, so the *_hr.dat and
# *_centres.xyz that TB2J reads have to come from the separate wannier90.x
# pass below.
vasp -f Wan90 -s {vasp_symmetry}

# There is no standalone wannier90 module on this cluster, but the Quantum
# ESPRESSO module bundles wannier90.x 3.1.0 -- the version the Wan90 VASP build
# was linked against. VASP's spin index 1 is up and 2 is down; the seednames
# are the PREFIX_UP / PREFIX_DN that the TB2J stage looks for.
module purge
module load {wannier90_module}

for pair in 1:up 2:dn; do
  seed="wannier90.${{pair#*:}}"
  # VASP writes a complete .win -- the settings from WANNIER90_WIN plus the
  # cell, atoms, k-points and num_bands it generated itself -- so the
  # standalone run reuses it verbatim. Reusing it is also what keeps num_bands
  # honest: VASP rounds NBANDS up to a multiple of the rank count, and the
  # matrices on disk have that many bands, not the number the INCAR asked for.
  cp wannier90.win "${{seed}}.win"
  # The matrices are large and read-only here, so link rather than copy.
  for ext in amn mmn eig; do
    ln -sf "wannier90.${{pair%%:*}}.${{ext}}" "${{seed}}.${{ext}}"
  done
  srun -n 1 wannier90.x "${{seed}}"
done
"""

TB2J_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=bw26b012
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir={output_dir}

# wann2J.py comes from this project's virtualenv, not from a cluster module,
# and it parallelises over the contour energy points with multiprocessing
# rather than MPI: a single task on one node with --np worker processes.
source {venv_activate}

{command}
"""