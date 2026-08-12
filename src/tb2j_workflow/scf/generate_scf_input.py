#!/usr/bin/env python
"""Write VASP SCF input files for a Materials Project structure.

First step of the TB2J workflow: fetch a structure from the Materials
Project by its material ID and write out INCAR/POSCAR/POTCAR/KPOINTS for a
static (SCF) calculation, built on top of pymatgen's MPStaticSet.

Requires a Materials Project API key, available via --api-key, the
MP_API_KEY environment variable, or pymatgen's config file (~/.config/.pmgrc.yaml).
"""

import argparse
from argparse import Namespace
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.io.vasp.sets import MPScanStaticSet, MPStaticSet, VaspInputSet
from pymatgen.core.structure import Structure

from ..sbatch_templates import SCF_TEMPLATE as SBATCH_TEMPLATE

def parse_incar_overrides(pairs: list[str]) -> dict:
    settings = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not _:
            raise argparse.ArgumentTypeError(f"invalid --incar entry {pair!r}, expected KEY=VALUE")
        key = key.strip().upper()
        value = value.strip()
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        else:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
        settings[key] = value
    return settings


def get_structure(mp_id: str, api_key: str | None, conventional_cell: bool) -> Structure:
    with MPRester(api_key) as mpr:
        return mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=conventional_cell)


def build_input_set(structure:Structure, args:Namespace) -> VaspInputSet:
    input_set_cls : VaspInputSet = MPScanStaticSet if args.functional == "r2scan" else MPStaticSet

    if args.magmom is not None:
        if len(args.magmom) != len(structure):
            raise ValueError(
                f"--magmom got {len(args.magmom)} values but structure has {len(structure)} sites"
            )
        structure.add_site_property("magmom", args.magmom)

    user_incar_settings = parse_incar_overrides(args.incar)

    return input_set_cls(
        structure=structure,
        user_kpoints_settings={"reciprocal_density": args.kpoints_density},
        user_incar_settings=user_incar_settings,
        user_potcar_functional=args.potcar_functional,
        auto_ispin=args.auto_ispin,
    )

def write_sbatch_script(output_dir: Path, args: argparse.Namespace) -> Path:
    script = SBATCH_TEMPLATE.format(
        job_name=args.job_name or args.mp_id,
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        time=args.time,
        vasp_module=args.vasp_module,
        vasp_executable=args.vasp_executable,
        output_dir=output_dir,
    )
    script_path = output_dir / "submit.sh"
    script_path.write_text(script)
    return script_path


def cli() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter) 
    parser.add_argument("mp_id", help="Materials Project ID, e.g. mp-19770")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory to write inputs to (default: ./work/<mp_id>/scf)")
    parser.add_argument("--api-key", default=None, help="Materials Project API key (default: MP_API_KEY env var / pmgrc config)")
    parser.add_argument("--functional", choices=["pbe", "r2scan"], default="pbe", help="Exchange-correlation functional / input set to use (default: pbe)")
    parser.add_argument("--potcar-functional", default="PBE_54", help="POTCAR functional flavor (default: PBE_54)")
    parser.add_argument("--kpoints-density", type=int, default=100, help="Reciprocal density for k-point mesh generation (default: 100)")
    parser.add_argument("--conventional-cell", action="store_true", help="Use the conventional standard cell instead of the primitive cell")
    parser.add_argument("--magmom", type=float, nargs="+", default=None, help="Explicit per-site initial magnetic moments, one value per atom in structure order")
    parser.add_argument("--auto-ispin", action="store_true", help="Let pymatgen switch ISPIN=1 if all MAGMOM values are ~0 after a previous run")
    parser.add_argument("--incar", nargs="*", default=[], metavar="KEY=VALUE", help="Additional/overriding INCAR settings, e.g. --incar ISYM=0 EDIFF=1e-7")
    parser.add_argument("--no-sbatch", action="store_true", help="Don't write a submit.sh sbatch script")
    parser.add_argument("--job-name", default=None, help="SLURM job name (default: <mp_id>)")
    parser.add_argument("--partition", default="standard", help="SLURM partition (default: standard)")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to request (default: 1)")
    parser.add_argument("--ntasks-per-node", type=int, default=24, help="MPI tasks per node (default: 24)")
    parser.add_argument("--time", default="00:10:00", help="Wall-clock time limit, SLURM format (default: 00:10:00)")
    parser.add_argument("--vasp-module", default="chem/vasp/6.4.3", help="Environment module to load for VASP")
    parser.add_argument("--vasp-executable", default="vasp_std", help="VASP executable to run (default: vasp_std)")
    args = parser.parse_args()
    return args

def main():
    args = cli()
    output_dir = args.output_dir or Path(f"./work/{args.mp_id}/scf")
    structure = get_structure(args.mp_id, args.api_key, args.conventional_cell)
    input_set = build_input_set(structure, args)
    input_set.write_input(str(output_dir), make_dir_if_not_present=True)
    print(f"Wrote SCF input files for {args.mp_id} to {output_dir}/")

    if not args.no_sbatch:
        script_path = write_sbatch_script(output_dir, args)
        print(f"Wrote sbatch script to {script_path}")


if __name__ == "__main__":
    main()
