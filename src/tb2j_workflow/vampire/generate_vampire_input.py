"""Write the Vampire run for a finished TB2J calculation.

Fourth step of the TB2J workflow: take the exchange parameters TB2J wrote as a
Vampire model and set up the atomistic spin dynamics run that turns them into
an ordering temperature. TB2J already writes ``TB2J_results/Vampire/`` with the
three files Vampire needs, so the model itself -- the unit cell with its
tensorial J(R) (``vampire.UCF``) and the per-sublattice moments
(``vampire.mat``) -- is copied over untouched. Only ``input``, the simulation
protocol, is written here, because that is the part TB2J fills in with generic
placeholders:

* ``sim:program=curie-temperature`` sweeps the temperature and records the
  magnetisation at each step, which is what the ordering temperature is read
  off. The sweep range and the equilibration/averaging lengths are flags.
* ``output:material-mean-magnetisation-length`` replaces TB2J's instantaneous
  ``output:material-magnetisation``. Vampire calls its output routine once per
  temperature, after the averaging loop, so the mean statistics give one row
  per temperature while the instantaneous ones give a single snapshot whose
  thermal noise is the size of the effect being measured.
* It is a *per-material* magnetisation because the sublattices are what stays
  finite below the ordering temperature. The moments TB2J writes all start
  along +z, so an antiferromagnetic set of J relaxes into its Neel state during
  equilibration and the total magnetisation goes to zero at every temperature;
  ``output:mean-magnetisation-length`` is kept alongside it precisely because
  that contrast is what says which of the two orderings the J describe.
* ``output:mean-susceptibility`` adds the four columns chi_x, chi_y, chi_z and
  chi_m, the fluctuation of the system magnetisation over the same averaging
  loop. Its peak is what the ``viz-results`` stage reads the ordering
  temperature off: M(T) only bends over near the transition, whereas chi
  diverges at it, so the peak is a much sharper estimator than any feature of
  the magnetisation curve. It is also the quantity the Curie-Weiss fit of
  1/chi extrapolates along, and that fit needs a paramagnetic tail reaching
  well past the transition -- a sweep that stops just above it has nothing to
  fit.

The cell repeat lengths come out of the UCF header rather than being recomputed
from the structure, so the geometry Vampire builds is the one the interactions
were tabulated for.
"""

import argparse
import shutil
from argparse import Namespace
from pathlib import Path

from ..sbatch_templates import VAMPIRE_TEMPLATE as SBATCH_TEMPLATE

UCF_NAME = "vampire.UCF"
MAT_NAME = "vampire.mat"

INPUT_TEMPLATE = """#------------------------------------------
# Vampire input for {mp_id}, written by gen-vampire.
# Exchange, moments and geometry from {source}
#------------------------------------------
# Creation attributes
#------------------------------------------
create:full
create:periodic-boundaries-x
create:periodic-boundaries-y
create:periodic-boundaries-z
#------------------------------------------
# The model TB2J wrote: unit cell with the tensorial J(R), and the moments
# and initial spin directions per sublattice.
#------------------------------------------
material:file = {mat_name}
material:unit-cell-file = "{ucf_name}"
dimensions:unit-cell-size-x = {a}
dimensions:unit-cell-size-y = {b}
dimensions:unit-cell-size-z = {c}
#------------------------------------------
# System dimensions
#------------------------------------------
dimensions:system-size-x = {system_size} !nm
dimensions:system-size-y = {system_size} !nm
dimensions:system-size-z = {system_size} !nm
#------------------------------------------
# Simulation attributes
#------------------------------------------
sim:program = curie-temperature
sim:integrator = llg-heun
sim:minimum-temperature = {temperature_min}
sim:maximum-temperature = {temperature_max}
sim:temperature-increment = {temperature_step}
sim:equilibration-time-steps = {equilibration_steps}
sim:loop-time-steps = {loop_steps}
# Sample the averages every step of the loop; the output itself is written
# once per temperature, after the loop, not once per increment.
sim:time-steps-increment = 1
#------------------------------------------
# Data output: one row per temperature -- the sublattice magnetisations first,
# then the magnetisation of the whole system, then
# the four susceptibility columns chi_x, chi_y, chi_z, chi_m.
#------------------------------------------
output:temperature
output:material-mean-magnetisation-length
output:mean-magnetisation-length
output:mean-susceptibility
"""


class TB2JError(RuntimeError):
    """The TB2J stage cannot be used as input for Vampire."""


def read_unitcell(ucf_path: Path) -> tuple[list[float], int, int]:
    """Cell lengths in Angstrom, number of spins and number of interactions."""
    lines = [line.strip() for line in ucf_path.read_text().splitlines() if line.strip()]

    def line_after(marker: str) -> str:
        for index, line in enumerate(lines[:-1]):
            if line.lower().startswith(marker):
                return lines[index + 1]
        raise TB2JError(
            f"{ucf_path} has no '{marker}' section, so it is not a Vampire unit cell file"
        )

    try:
        cell = [float(length) for length in line_after("# unit cell size").split()]
        spins = int(line_after("# atoms").split()[0])
        interactions = int(line_after("# interactions").split()[0])
    except (IndexError, ValueError) as error:
        raise TB2JError(
            f"{ucf_path} is not readable as a Vampire unit cell file: {error}"
        ) from error

    if len(cell) != 3:
        raise TB2JError(f"{ucf_path} gives {len(cell)} unit cell lengths instead of 3")
    if spins < 1 or interactions < 1:
        raise TB2JError(
            f"{ucf_path} has {spins} spins and {interactions} interactions: the TB2J run found no "
            f"exchange to simulate. Check {ucf_path.parent.parent}/exchange.out"
        )
    return cell, spins, interactions


def read_num_materials(mat_path: Path) -> int:
    """Sublattices in the material file, one per spin of the unit cell."""
    for line in mat_path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.replace(" ", "") == "material:num-materials":
            try:
                return int(value)
            except ValueError as error:
                raise TB2JError(
                    f"{mat_path} has an unreadable num-materials: {error}"
                ) from error
    raise TB2JError(
        f"{mat_path} has no 'material:num-materials' line, so it is not a Vampire material file"
    )


def read_tb2j_model(source_dir: Path) -> tuple[list[float], int, int]:
    if not source_dir.is_dir():
        raise TB2JError(
            f"{source_dir} does not exist: run the TB2J stage first, it writes the Vampire model "
            f"alongside its own results"
        )
    for name in (UCF_NAME, MAT_NAME):
        if not (source_dir / name).is_file():
            raise TB2JError(
                f"{source_dir}/{name} is missing, so the TB2J run did not finish"
            )

    cell, spins, interactions = read_unitcell(source_dir / UCF_NAME)
    materials = read_num_materials(source_dir / MAT_NAME)
    if materials != spins:
        raise TB2JError(
            f"{source_dir} is inconsistent: {UCF_NAME} has {spins} spins but {MAT_NAME} defines "
            f"{materials} materials"
        )
    return cell, spins, interactions


def build_input(
    mp_id: str, source_dir: Path, cell: list[float], args: Namespace
) -> str:
    return INPUT_TEMPLATE.format(
        mp_id=mp_id,
        source=source_dir.resolve(),
        mat_name=MAT_NAME,
        ucf_name=UCF_NAME,
        a=f"{cell[0]:.6f}",
        b=f"{cell[1]:.6f}",
        c=f"{cell[2]:.6f}",
        system_size=args.system_size,
        temperature_min=args.temperature_min,
        temperature_max=args.temperature_max,
        temperature_step=args.temperature_step,
        equilibration_steps=args.equilibration_steps,
        loop_steps=args.loop_steps,
    )


def system_cells(cell: list[float], system_size: float) -> list[int]:
    """Whole unit cells Vampire fits along each axis of the requested box."""
    return [max(1, int(system_size * 10.0 / length)) for length in cell]


def write_sbatch_script(output_dir: Path, args: Namespace) -> Path:
    binary = Path(args.vampire_binary)
    if not binary.is_file():
        raise TB2JError(
            f"no vampire binary at {binary}: point --vampire-binary at the build to run "
            f"(or use --no-sbatch)"
        )

    script = SBATCH_TEMPLATE.format(
        job_name=args.job_name or f"{args.mp_id}-vampire",
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        time=args.time,
        output_dir=output_dir.resolve(),
        modules=args.modules,
        vampire_binary=binary.resolve(),
    )
    script_path = output_dir / "submit.sh"
    script_path.write_text(script)
    return script_path


def cli() -> Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mp_id", help="Materials Project ID, e.g. mp-19770")
    parser.add_argument(
        "--tb2j-dir",
        type=Path,
        default=None,
        help="Directory holding the finished TB2J run (default: ./work/<mp_id>/tb2j)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to run Vampire in (default: ./work/<mp_id>/vampire)",
    )
    parser.add_argument(
        "--system-size",
        type=float,
        default=15.0,
        help="Edge of the cubic simulation box in nm, filled with whole unit cells (default: 15.0)",
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=0.0,
        help="Lowest temperature of the sweep in K (default: 0.0)",
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=1000.0,
        help="Highest temperature of the sweep in K (default: 1000.0)",
    )
    parser.add_argument(
        "--temperature-step",
        type=float,
        default=25.0,
        help="Temperature increment of the sweep in K (default: 25.0)",
    )
    parser.add_argument(
        "--equilibration-steps",
        type=int,
        default=2500,
        help="Time steps to equilibrate at each temperature before averaging (default: 2500)",
    )
    parser.add_argument(
        "--loop-steps",
        type=int,
        default=3000,
        help="Time steps to average over at each temperature (default: 3000)",
    )
    parser.add_argument(
        "--no-sbatch", action="store_true", help="Don't write a submit.sh sbatch script"
    )
    parser.add_argument(
        "--job-name", default=None, help="SLURM job name (default: <mp_id>-vampire)"
    )
    parser.add_argument(
        "--partition", default="standard", help="SLURM partition (default: standard)"
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes to request (default: 1)"
    )
    parser.add_argument(
        "--ntasks-per-node",
        type=int,
        default=24,
        help="MPI tasks per node (default: 24)",
    )
    parser.add_argument(
        "--time",
        default="04:00:00",
        help="Wall-clock time limit, SLURM format (default: 04:00:00)",
    )
    parser.add_argument(
        "--vampire-binary",
        default="/lustre/work/ws/ws1/ka_qa7606-dft_pipeline/vampire/vampire-parallel",
        help="Vampire executable to run under srun (default: the vampire-parallel build in the DFT pipeline workspace)",
    )
    parser.add_argument(
        "--modules",
        default="compiler/intel/2024.2.1 mpi/impi/2021.13.1",
        help="Environment modules to load, matching the MPI the binary was linked against (default: compiler/intel/2024.2.1 mpi/impi/2021.13.1)",
    )
    return parser.parse_args()


def main():
    args = cli()
    tb2j_dir = args.tb2j_dir or Path(f"./work/{args.mp_id}/tb2j")
    source_dir = tb2j_dir / "TB2J_results" / "Vampire"
    output_dir = args.output_dir or Path(f"./work/{args.mp_id}/vampire")

    cell, spins, interactions = read_tb2j_model(source_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (UCF_NAME, MAT_NAME):
        shutil.copy(source_dir / name, output_dir / name)
    (output_dir / "input").write_text(build_input(args.mp_id, source_dir, cell, args))

    cells = system_cells(cell, args.system_size)
    temperatures = (
        int((args.temperature_max - args.temperature_min) // args.temperature_step) + 1
    )

    print(f"Wrote the Vampire stage for {args.mp_id} to {output_dir}/")
    print(
        f"  model:         {source_dir}/ ({spins} spins per cell, {interactions} interactions)"
    )
    print(f"  unit cell:     {' '.join(f'{length:.4f}' for length in cell)} A")
    print(
        f"  system:        {args.system_size} nm cube, {' x '.join(str(n) for n in cells)} cells, "
        f"{cells[0] * cells[1] * cells[2] * spins} spins"
    )
    print(
        f"  sweep:         {args.temperature_min} to {args.temperature_max} K in steps of "
        f"{args.temperature_step} K ({temperatures} temperatures)"
    )
    print(
        f"  per point:     {args.equilibration_steps} equilibration + {args.loop_steps} averaging steps"
    )
    print(f"  results:       {output_dir}/output")

    if not args.no_sbatch:
        script_path = write_sbatch_script(output_dir, args)
        print(f"Wrote sbatch script to {script_path}")


if __name__ == "__main__":
    main()
