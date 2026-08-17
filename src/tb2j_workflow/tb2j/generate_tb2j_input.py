#!/usr/bin/env python
"""Write the TB2J run for a finished VASP+Wannier90 calculation.

Third step of the TB2J workflow: take the wannierisation directory and set up
the ``wann2J.py`` run that turns the two spin channels of the real-space
Wannier Hamiltonian into Heisenberg exchange parameters via the magnetic force
theorem. No inputs of its own are written: TB2J reads the Wannier files in
place, so this stage only produces the sbatch script that calls ``wann2J.py``
with arguments derived from the wannierisation directory.

Everything wann2J.py needs is taken from that directory, so nothing has to be
chosen per material:

* ``--efermi`` comes from the NSCF ``vasprun.xml``. That is the run whose
  eigenvalues the Wannier Hamiltonian was fitted to, so its Fermi level is the
  one consistent with the Hamiltonian TB2J reads (the SCF value is the same
  number, since the NSCF run reuses the SCF density on the same mesh).
* ``--elements`` are the elements carrying a moment in the INCAR MAGMOM, i.e.
  the sites the magnetic force theorem is applied to.
* ``--kmesh`` is the DFT mesh from KPOINTS. The Hamiltonian only carries the
  real-space vectors inside the Wannier supercell that this mesh defines, so a
  denser mesh would produce J(R) at vectors where the Hamiltonian is padding
  rather than data - long-range tails that look like physics but are not.
* ``--prefix_up`` / ``--prefix_down`` are the ``wannier90.up`` /
  ``wannier90.dn`` files VASP writes for an ISPIN=2 wannierisation.
"""

import argparse
import shlex
import sys
from argparse import Namespace
from pathlib import Path

from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Vasprun

from ..sbatch_templates import TB2J_TEMPLATE as SBATCH_TEMPLATE

# Prefixes VASP gives the per-spin Wannier files of an ISPIN=2 run with
# LWANNIER90=.TRUE.; they are also wann2J.py's defaults.
PREFIX_UP = "wannier90.up"
PREFIX_DN = "wannier90.dn"


class WannierError(RuntimeError):
    """The wannierisation stage cannot be used as input for TB2J."""


def read_wannier_run(wannier_dir: Path) -> tuple[Incar, Poscar, Kpoints]:
    for name in ("INCAR", "POSCAR", "KPOINTS", "vasprun.xml"):
        if not (wannier_dir / name).is_file():
            raise WannierError(f"{wannier_dir}/{name} is missing, run the wannierisation stage first")

    for prefix in (PREFIX_UP, PREFIX_DN):
        # HamiltonIO reads the Hamiltonian from _tb.dat if it is there and
        # falls back to _hr.dat, and needs the centres to place the orbitals.
        if not any((wannier_dir / f"{prefix}_{suffix}.dat").is_file() for suffix in ("tb", "hr")):
            raise WannierError(
                f"{wannier_dir}/{prefix}_hr.dat is missing: the wannierisation did not write a "
                f"Hamiltonian for this spin channel. Check wannier90.wout for why it stopped"
            )
        if not (wannier_dir / f"{prefix}_centres.xyz").is_file():
            raise WannierError(
                f"{wannier_dir}/{prefix}_centres.xyz is missing, the .win block needs write_xyz = .true."
            )

    return (
        Incar.from_file(wannier_dir / "INCAR"),
        Poscar.from_file(wannier_dir / "POSCAR"),
        Kpoints.from_file(wannier_dir / "KPOINTS"),
    )


def read_efermi(wannier_dir: Path) -> float:
    """Fermi level of the NSCF run that produced the Wannier functions."""
    vasprun = Vasprun(
        str(wannier_dir / "vasprun.xml"),
        parse_dos=True,  # what fills in vasprun.efermi
        parse_eigen=False,
        parse_potcar_file=False,
    )
    if vasprun.efermi is None:
        raise WannierError(f"{wannier_dir}/vasprun.xml has no Fermi energy, pass one with --efermi")
    return float(vasprun.efermi)


def magnetic_elements(incar: Incar, poscar: Poscar, threshold: float) -> list[str]:
    """Elements that start out with a moment, in POSCAR order.

    The threshold has to sit well above pymatgen's generic 0.6 muB seed, which
    every site gets whether or not it is magnetic, and below the 5 muB it seeds
    the magnetic transition metals with.
    """
    if int(incar.get("ISPIN", 1)) != 2:
        raise WannierError(
            "the wannierisation ran with ISPIN=1, so there is no spin splitting to extract "
            "exchange parameters from"
        )

    magmom = incar.get("MAGMOM")
    if magmom is None:
        raise WannierError("the wannierisation INCAR has no MAGMOM, so the magnetic sites are unknown")

    symbols = [site.specie.symbol for site in poscar.structure]
    if len(magmom) != len(symbols):
        raise WannierError(
            f"MAGMOM has {len(magmom)} entries for {len(symbols)} sites; only collinear runs "
            f"(one moment per site) are supported"
        )

    elements: list[str] = []
    for symbol, moment in zip(symbols, magmom, strict=True):
        if abs(float(moment)) >= threshold and symbol not in elements:
            elements.append(symbol)
    if not elements:
        raise WannierError(
            f"no element has a MAGMOM of at least {threshold} muB, so there are no magnetic sites; "
            f"lower --magmom-threshold or set --elements explicitly"
        )
    return elements


def dft_kmesh(kpoints: Kpoints) -> list[int]:
    """The Gamma/Monkhorst mesh divisions the DFT runs used."""
    if kpoints.style not in (Kpoints.supported_modes.Gamma, Kpoints.supported_modes.Monkhorst):
        raise WannierError(
            f"the KPOINTS file is in {kpoints.style} mode, which has no mesh to reuse; "
            f"pass one with --kmesh"
        )
    return [int(divisions) for divisions in kpoints.kpts[0]]


def build_description(mp_id: str, incar: Incar, poscar: Poscar, elements: list[str]) -> str:
    """One line of provenance for TB2J to record in its output."""
    functional = "r2SCAN" if str(incar.get("METAGGA", "")).lower() == "r2scan" else "PBE"
    parts = [f"{mp_id}: VASP {functional}"]

    if incar.get("LDAU") and incar.get("LDAUU"):
        hubbard = [
            f"{symbol} U={value}"
            for symbol, value in zip(poscar.site_symbols, incar["LDAUU"], strict=False)
            if float(value) != 0.0
        ]
        if hubbard:
            parts[0] += "+U (" + ", ".join(hubbard) + ")"

    magmom = " ".join(f"{float(moment):g}" for moment in incar["MAGMOM"])
    parts.append(f"collinear ISPIN=2, MAGMOM = {magmom}")
    parts.append(f"exchange on {' '.join(elements)} from a VASP/Wannier90 Hamiltonian")
    return ", ".join(parts)


def build_command(
    wannier_dir: Path, elements: list[str], kmesh: list[int], efermi: float, description: str, args: Namespace
) -> str:
    # --elements and --kmesh take several values each, so every value is a
    # shell word of its own; quoting the joined string would hand wann2J.py a
    # single element literally named "Fe O".
    arguments: list[tuple[str, list[str]]] = [
        # --posfile is resolved relative to --path, so the wannierisation
        # POSCAR is what fixes the structure TB2J sees.
        ("--path", [str(wannier_dir.resolve())]),
        ("--posfile", ["POSCAR"]),
        ("--prefix_up", [PREFIX_UP]),
        ("--prefix_down", [PREFIX_DN]),
        ("--elements", elements),
        ("--kmesh", [str(divisions) for divisions in kmesh]),
        ("--efermi", [f"{efermi:.6f}"]),
        ("--emin", [str(args.emin)]),
        ("--emax", [str(args.emax)]),
        ("--nz", [str(args.nz)]),
        ("--np", [str(args.cpus)]),
        ("--description", [description]),
    ]
    if args.rcut is not None:
        arguments.append(("--rcut", [str(args.rcut)]))

    lines = [
        " ".join([flag, *(shlex.quote(value) for value in values)]) for flag, values in arguments
    ]
    return "wann2J.py \\\n" + " \\\n".join(f"    {line}" for line in lines)


def write_sbatch_script(output_dir: Path, command: str, args: Namespace) -> Path:
    activate = Path(sys.prefix) / "bin" / "activate"
    if not activate.is_file():
        raise WannierError(
            f"no virtualenv activate script at {activate}: the sbatch script needs one to put "
            f"wann2J.py on PATH, so run gen-tb2j from the project venv (or use --no-sbatch)"
        )

    script = SBATCH_TEMPLATE.format(
        job_name=args.job_name or f"{args.mp_id}-tb2j",
        partition=args.partition,
        cpus=args.cpus,
        time=args.time,
        output_dir=output_dir.resolve(),
        venv_activate=activate,
        command=command,
    )
    script_path = output_dir / "submit.sh"
    script_path.write_text(script)
    return script_path


def cli() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mp_id", help="Materials Project ID, e.g. mp-19770")
    parser.add_argument("--wannier-dir", type=Path, default=None, help="Directory holding the finished Wannier90 run (default: ./work/<mp_id>/wannier90)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory to run TB2J in (default: ./work/<mp_id>/tb2j)")
    parser.add_argument("--elements", nargs="+", default=None, metavar="EL", help="Override the MAGMOM-derived magnetic elements, e.g. --elements Fe")
    parser.add_argument("--magmom-threshold", type=float, default=1.0, help="Smallest MAGMOM in muB for an element to count as magnetic; above pymatgen's 0.6 default seed (default: 1.0)")
    parser.add_argument("--kmesh", type=int, nargs=3, default=None, metavar=("NX", "NY", "NZ"), help="Override the k-mesh for the exchange integration (default: the DFT mesh from KPOINTS)")
    parser.add_argument("--efermi", type=float, default=None, help="Fermi energy in eV (default: the value from the Wannier90 run's vasprun.xml)")
    parser.add_argument("--emin", type=float, default=-14.0, help="Lower end of the energy contour, relative to efermi (default: -14.0)")
    parser.add_argument("--emax", type=float, default=0.0, help="Upper end of the energy contour, relative to efermi (default: 0.0)")
    parser.add_argument("--nz", type=int, default=100, help="Number of points on the semicircle contour (default: 100)")
    parser.add_argument("--rcut", type=float, default=None, help="Cutoff in Angstrom on the spin-pair distance (default: every R commensurate with the k-mesh)")
    parser.add_argument("--no-sbatch", action="store_true", help="Don't write a submit.sh sbatch script")
    parser.add_argument("--job-name", default=None, help="SLURM job name (default: <mp_id>-tb2j)")
    parser.add_argument("--partition", default="standard", help="SLURM partition (default: standard)")
    parser.add_argument("--cpus", type=int, default=24, help="Worker processes for wann2J.py, requested as cpus-per-task (default: 24)")
    parser.add_argument("--time", default="02:00:00", help="Wall-clock time limit, SLURM format (default: 02:00:00)")
    return parser.parse_args()


def main():
    args = cli()
    wannier_dir = args.wannier_dir or Path(f"./work/{args.mp_id}/wannier90")
    output_dir = args.output_dir or Path(f"./work/{args.mp_id}/tb2j")

    incar, poscar, kpoints = read_wannier_run(wannier_dir)

    elements = args.elements or magnetic_elements(incar, poscar, args.magmom_threshold)
    kmesh = args.kmesh or dft_kmesh(kpoints)
    efermi = args.efermi if args.efermi is not None else read_efermi(wannier_dir)
    description = build_description(args.mp_id, incar, poscar, elements)

    command = build_command(wannier_dir, elements, kmesh, efermi, description, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Wrote the TB2J stage for {args.mp_id} to {output_dir}/")
    print(f"  wannier files: {wannier_dir}/{PREFIX_UP}, {wannier_dir}/{PREFIX_DN}")
    print(f"  elements:      {' '.join(elements)}")
    print(f"  kmesh:         {' '.join(str(divisions) for divisions in kmesh)}")
    print(f"  efermi:        {efermi:.4f} eV")
    print(f"  contour:       {args.emin} to {args.emax} eV around efermi, {args.nz} points")
    print(f"  results:       {output_dir}/TB2J_results/")

    if not args.no_sbatch:
        script_path = write_sbatch_script(output_dir, command, args)
        print(f"Wrote sbatch script to {script_path}")


if __name__ == "__main__":
    main()
