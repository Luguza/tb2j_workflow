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

module load {vasp_module}

# -f Wan90 selects the VASP build linked against the wannier90 library, so
# LWANNIER90=.TRUE. runs the wannierisation in-process and writes *_hr.dat.
vasp -f Wan90 -s {vasp_symmetry}
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