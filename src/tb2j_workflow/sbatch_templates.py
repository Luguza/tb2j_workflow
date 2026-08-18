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

# -f Wan90 selects the VASP build linked against the wannier90 library, which
# LWANNIER90_RUN drives all the way through: VASP calls wannier_setup and then
# wannier_run once per spin channel, writing the *_hr.dat and *_centres.xyz
# that TB2J reads under the per-spin seednames wannier90.1 (up) and
# wannier90.2 (down). The plain vasp binary would silently ignore both tags.
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

VAMPIRE_TEMPLATE = """#!/bin/bash
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

# vampire-parallel is a local build rather than a cluster module, so the
# modules loaded here are the ones it was linked against -- srun launches it
# through that MPI's PMI, and a mismatched libmpi fails at startup.
module load {modules}

# vampire reads `input`, `vampire.mat` and `vampire.UCF` from the working
# directory (no arguments) and writes `output` and `log` next to them.
srun {vampire_binary}
"""
