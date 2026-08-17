#!/usr/bin/env python
"""Write VASP+Wannier90 input files for a finished SCF calculation.

Second step of the TB2J workflow: take the output of the SCF stage and set up
a non-self-consistent run (ICHARG=11, restarting from the SCF CHGCAR) with
LWANNIER90=.TRUE., so that VASP builds the maximally localised Wannier
functions in-process and writes the ``*_hr.dat`` / ``*_centres.xyz`` files
that TB2J's ``wann2J.py`` reads.

Everything is derived from the SCF directory so that no per-material input is
needed:

* INCAR settings (ENCUT, LDAU*, MAGMOM, ISPIN, ...) are inherited from the SCF
  INCAR; POSCAR/POTCAR/KPOINTS are reused verbatim and the CHGCAR is symlinked.
* The projections are every valence orbital carried by the POTCAR of each
  element (e.g. ``Fe_pv -> s;p;d``, ``O -> s;p``), which fixes num_wann.
* The disentanglement windows are computed from the SCF eigenvalues. Because
  the NSCF run reuses the SCF charge density on the same k-mesh, its
  eigenvalues are the SCF ones (the full-grid k-points are symmetry images of
  the IBZ k-points), so the windows can be chosen to satisfy Wannier90's two
  hard constraints exactly rather than heuristically:

  - ``dis_froz_max = min_k E_{num_wann+1}(k) - margin`` keeps at most num_wann
    states frozen at every k, and is the largest window that does so.
  - ``dis_win_max  = min_k E_{nbands-top_margin}(k) - margin`` keeps at least
    num_wann states inside the outer window at every k.

  Both are taken as the minimum over spin channels, since VASP applies one
  WANNIER90_WIN block to both.
"""

import argparse
import os
import shutil
from argparse import Namespace
from pathlib import Path

from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar
from pymatgen.io.vasp.outputs import Vasprun

from ..sbatch_templates import WANNIER90_TEMPLATE as SBATCH_TEMPLATE
from ..scf.generate_scf_input import parse_incar_overrides
from .plot_dos_windows import plot_dos_windows
from .plot_projectability import (
    band_projectability,
    plot_projectability,
    projectability_summary,
)

# Orbitals per angular momentum channel, in the order Wannier90 expects them
# to be listed in a projection line.
ORBITALS = {"s": 1, "p": 3, "d": 5, "f": 7}

# INCAR keys that belong to the SCF stage and must not leak into the NSCF run.
SCF_ONLY_INCAR_KEYS = ("NSW", "IBRION", "ISIF", "LAECHG", "LVTOT", "LVHAR", "LELF")


class ScfError(RuntimeError):
    """The SCF stage cannot be used as input for the wannierisation."""


def read_scf(scf_dir: Path) -> tuple[Incar, Poscar, Potcar, Vasprun]:
    for name in ("INCAR", "POSCAR", "POTCAR", "KPOINTS", "CHGCAR", "vasprun.xml"):
        if not (scf_dir / name).is_file():
            raise ScfError(f"{scf_dir}/{name} is missing, run the SCF stage first")

    # parse_dos stays on: it is what fills in vasprun.efermi, which the TB2J
    # stage needs. The projected eigenvalues feed the projectability check and
    # cost a fraction of a second on top of the rest of the parse.
    vasprun = Vasprun(
        str(scf_dir / "vasprun.xml"),
        parse_projected_eigen=True,
        parse_potcar_file=False,
    )
    if not vasprun.converged_electronic:
        raise ScfError(f"the SCF run in {scf_dir} is not electronically converged")

    return (
        Incar.from_file(scf_dir / "INCAR"),
        Poscar.from_file(scf_dir / "POSCAR"),
        Potcar.from_file(scf_dir / "POTCAR"),
        vasprun,
    )


def count_wannier_functions(poscar: Poscar, projections: list[str]) -> int:
    """num_wann implied by a list of ``El:orb;orb`` projection lines."""
    counts = dict(zip(poscar.site_symbols, poscar.natoms, strict=True))
    num_wann = 0
    for line in projections:
        element, _, orbitals = line.partition(":")
        if element not in counts:
            raise ScfError(f"projection {line!r} names element {element!r}, which is not in the POSCAR")
        num_wann += counts[element] * sum(ORBITALS[orbital] for orbital in orbitals.split(";"))
    return num_wann


def valence_projections(poscar: Poscar, potcar: Potcar) -> tuple[list[str], int, int]:
    """Project onto every valence orbital the POTCARs carry.

    Returns the Wannier90 projection lines, the resulting num_wann, and the
    number of semicore bands to exclude. Using the full valence set rather than
    a hand-picked d/p subset means no per-material choice has to be made, and
    the Wannier manifold spans the whole valence + low-lying conduction space.

    Where a POTCAR carries two shells of the same angular momentum (Sr_sv has
    4s and 5s, La has 5s and 6s), Wannier90's ``El:s`` syntax can only describe
    one of them, so the deeper shell is dropped from the projections. Those
    semicore bands are chemically inert but would still sit inside the
    disentanglement window, so they are counted here and excluded by band index
    instead.
    """
    counts = dict(zip(poscar.site_symbols, poscar.natoms, strict=True))
    if len(counts) != len(potcar):
        raise ScfError("POSCAR element blocks do not line up with the POTCAR")

    lines: list[str] = []
    num_wann = 0
    num_excluded = 0
    for element, potcar_single in zip(counts, potcar, strict=True):
        if potcar_single.element != Element(element).symbol:
            raise ScfError(
                f"POSCAR element {element} does not match POTCAR entry {potcar_single.symbol}"
            )

        # Keep the outermost shell of each angular momentum, drop deeper ones.
        outermost: dict[str, int] = {}
        dropped = 0
        for shell_n, orbital, _ in potcar_single.electron_configuration:
            if orbital not in ORBITALS:
                raise ScfError(f"unsupported valence orbital {orbital!r} in the POTCAR for {element}")
            if orbital in outermost:
                dropped += ORBITALS[orbital]
                if shell_n <= outermost[orbital]:
                    continue
            outermost[orbital] = shell_n

        orbitals = [orbital for orbital in ORBITALS if orbital in outermost]
        if not orbitals:
            raise ScfError(f"no valence orbitals found in the POTCAR for {element}")
        lines.append(f"{element}:{';'.join(orbitals)}")
        num_wann += counts[element] * sum(ORBITALS[orbital] for orbital in orbitals)
        num_excluded += counts[element] * dropped
    return lines, num_wann, num_excluded


def band_minimum(vasprun: Vasprun, band: int) -> float:
    """Lowest energy of the 1-based band index over all k-points and spins."""
    return min(
        float(eigenvalues[:, band - 1, 0].min()) for eigenvalues in vasprun.eigenvalues.values()
    )


def band_maximum(vasprun: Vasprun, band: int) -> float:
    """Highest energy of the 1-based band index over all k-points and spins."""
    return max(
        float(eigenvalues[:, band - 1, 0].max()) for eigenvalues in vasprun.eigenvalues.values()
    )


def disentanglement_windows(
    vasprun: Vasprun, num_wann: int, num_excluded: int, top_band_margin: int, margin: float
) -> tuple[float, float]:
    """Windows that Wannier90 is guaranteed to accept at every k-point.

    Wannier90 aborts if any k-point has more than num_wann states frozen, or
    fewer than num_wann states inside the outer window. Both are enforced here
    from the eigenvalues rather than guessed. Band indices are counted from
    num_excluded, since those lowest bands are excluded from the wannierisation.
    """
    nbands = int(vasprun.parameters["NBANDS"])
    # VASP leaves its topmost bands poorly converged, so prefer to keep them
    # out of the outer window.
    top_band = nbands - top_band_margin
    highest = num_excluded + num_wann

    if top_band < highest + 2:
        needed = highest + 2 + top_band_margin
        raise ScfError(
            f"the SCF run has NBANDS={nbands}, which leaves only {max(top_band - num_excluded, 0)} "
            f"usable bands for {num_wann} Wannier functions. Re-run the SCF stage with at least "
            f"NBANDS={needed}, e.g. gen-scf <mp-id> --incar NBANDS={needed}"
        )

    # Largest frozen window that can never over-fill: the band above the
    # manifold stays out at every k, so at most num_wann states are frozen.
    dis_froz_max = band_minimum(vasprun, highest + 1) - margin

    # The outer window must hold at least num_wann states at every k, i.e. sit
    # above the top of the manifold everywhere. Where the bands are dispersive
    # enough that this is higher than the top-band cut, the constraint wins and
    # some of the less well converged bands come along; that costs some quality
    # but is the only legal choice.
    dis_win_max = max(
        band_minimum(vasprun, top_band) - margin,
        band_maximum(vasprun, highest) + margin,
        dis_froz_max + margin,
    )
    return dis_froz_max, dis_win_max


def build_win_block(
    projections: list[str],
    num_wann: int,
    num_excluded: int,
    dis_froz_max: float,
    dis_win_max: float,
    args: Namespace,
) -> str:
    """The wannier90 .win body that VASP passes through to the library."""
    lines = [
        f"num_wann = {num_wann}",
        *([f"exclude_bands = 1-{num_excluded}"] if num_excluded else []),
        f"dis_win_max = {dis_win_max:.6f}",
        f"dis_froz_max = {dis_froz_max:.6f}",
        f"num_iter = {args.num_iter}",
        f"dis_num_iter = {args.dis_num_iter}",
        # Materials Project lattice vectors carry round-off (~1e-5 Ang) that
        # breaks the symmetry between axes that should be equivalent, which
        # splits a b-vector shell into near-degenerate pieces. Wannier90 counts
        # shell multiplicities with an absolute kmesh_tol but collects the
        # b-vectors with a relative one, so at the 1e-6 default the two
        # disagree and the shell search dies with "Not enough bvectors found".
        # 1e-4 merges the split consistently and is still far below the real
        # shell spacing.
        "kmesh_tol = 1e-4",
        # TB2J reads the real-space Hamiltonian and the Wannier centres.
        "write_hr = .true.",
        "write_xyz = .true.",
        "guiding_centres = .true.",
        "begin projections",
        *projections,
        "end projections",
    ]
    body = "\n".join(f"  {line}" for line in lines)
    return f'WANNIER90_WIN = "\n{body}\n"\n'


def build_incar(scf_incar: Incar, nbands: int, args: Namespace) -> Incar:
    incar = Incar(scf_incar.copy())
    for key in SCF_ONLY_INCAR_KEYS:
        incar.pop(key, None)
    incar.update(
        {
            "ICHARG": 11,  # non-self-consistent, read the SCF CHGCAR
            "ISYM": -1,  # Wannier90 needs the full, unsymmetrised k-grid
            "ALGO": "Normal",
            "NBANDS": nbands,
            "NELM": args.nelm,
            "EDIFF": args.ediff,
            "LWANNIER90": True,
            "LWAVE": False,
            "LCHARG": False,
        }
    )
    incar.update(parse_incar_overrides(args.incar))
    return incar


def write_inputs(scf_dir: Path, output_dir: Path, incar: Incar, win_block: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "INCAR").write_text(str(incar).rstrip("\n") + "\n" + win_block)
    for name in ("POSCAR", "POTCAR", "KPOINTS"):
        shutil.copy(scf_dir / name, output_dir / name)

    # The CHGCAR is large and read-only for an ICHARG=11 run, so link it.
    chgcar = output_dir / "CHGCAR"
    if chgcar.is_symlink() or chgcar.exists():
        chgcar.unlink()
    chgcar.symlink_to(os.path.relpath(scf_dir / "CHGCAR", output_dir))


def write_sbatch_script(output_dir: Path, args: Namespace) -> Path:
    script = SBATCH_TEMPLATE.format(
        job_name=args.job_name or f"{args.mp_id}-w90",
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        time=args.time,
        vasp_module=args.vasp_module,
        wannier90_module=args.wannier90_module,
        vasp_symmetry=args.vasp_symmetry,
        output_dir=output_dir.resolve(),
    )
    script_path = output_dir / "submit.sh"
    script_path.write_text(script)
    return script_path


def cli() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mp_id", help="Materials Project ID, e.g. mp-19770")
    parser.add_argument("--scf-dir", type=Path, default=None, help="Directory holding the finished SCF run (default: ./work/<mp_id>/scf)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory to write inputs to (default: ./work/<mp_id>/wannier90)")
    parser.add_argument("--projections", nargs="+", default=None, metavar="EL:ORBS", help="Override the POTCAR-derived projections, e.g. --projections Fe:d O:p")
    parser.add_argument("--num-iter", type=int, default=5000, help="Wannier90 localisation iterations (default: 5000)")
    parser.add_argument("--dis-num-iter", type=int, default=5000, help="Wannier90 disentanglement iterations (default: 5000)")
    parser.add_argument("--top-band-margin", type=int, default=4, help="Number of topmost VASP bands to keep out of the outer window (default: 4)")
    parser.add_argument("--window-margin", type=float, default=0.01, help="Safety margin in eV subtracted from both disentanglement windows (default: 0.01)")
    parser.add_argument("--nbands", type=int, default=None, help="NBANDS for the NSCF run (default: the value the SCF run used)")
    parser.add_argument("--ediff", type=float, default=1e-6, help="EDIFF for the NSCF run (default: 1e-6)")
    parser.add_argument("--nelm", type=int, default=200, help="NELM for the NSCF run (default: 200)")
    parser.add_argument("--incar", nargs="*", default=[], metavar="KEY=VALUE", help="Additional/overriding INCAR settings, e.g. --incar ISMEAR=0")
    parser.add_argument("--no-sbatch", action="store_true", help="Don't write a submit.sh sbatch script")
    parser.add_argument("--no-plot", action="store_true", help="Don't write the dos_windows.png check plot")
    parser.add_argument("--job-name", default=None, help="SLURM job name (default: <mp_id>-w90)")
    parser.add_argument("--partition", default="standard", help="SLURM partition (default: standard)")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to request (default: 1)")
    parser.add_argument("--ntasks-per-node", type=int, default=24, help="MPI tasks per node (default: 24)")
    parser.add_argument("--time", default="02:00:00", help="Wall-clock time limit, SLURM format (default: 02:00:00)")
    parser.add_argument("--vasp-module", default="chem/vasp/6.4.3", help="Environment module to load for VASP")
    parser.add_argument("--wannier90-module", default="chem/quantum_espresso/7.1", help="Environment module providing wannier90.x (default: chem/quantum_espresso/7.1)")
    parser.add_argument("--vasp-symmetry", choices=["std", "gam", "ncl"], default="std", help="VASP binary flavour to run (default: std)")
    return parser.parse_args()


def main():
    args = cli()
    scf_dir = args.scf_dir or Path(f"./work/{args.mp_id}/scf")
    output_dir = args.output_dir or Path(f"./work/{args.mp_id}/wannier90")

    scf_incar, poscar, potcar, vasprun = read_scf(scf_dir)

    projections, num_wann, num_excluded = valence_projections(poscar, potcar)
    if args.projections is not None:
        projections = args.projections
        num_wann = count_wannier_functions(poscar, projections)
        num_excluded = 0

    dis_froz_max, dis_win_max = disentanglement_windows(
        vasprun, num_wann, num_excluded, args.top_band_margin, args.window_margin
    )
    nbands = args.nbands or int(vasprun.parameters["NBANDS"])

    incar = build_incar(scf_incar, nbands, args)
    win_block = build_win_block(
        projections, num_wann, num_excluded, dis_froz_max, dis_win_max, args
    )
    write_inputs(scf_dir, output_dir, incar, win_block)

    print(f"Wrote Wannier90 input files for {args.mp_id} to {output_dir}/")
    print(f"  projections:  {' '.join(projections)}")
    print(f"  num_wann:     {num_wann} (NBANDS={nbands})")
    if num_excluded:
        print(f"  excluded:     {num_excluded} semicore bands (exclude_bands = 1-{num_excluded})")
    print(f"  dis_froz_max: {dis_froz_max:.4f} eV   (efermi={vasprun.efermi:.4f} eV)")
    print(f"  dis_win_max:  {dis_win_max:.4f} eV")

    # Diagnostics, so an SCF run without LORBIT must not stop the inputs from
    # being written.
    try:
        energies, projectabilities = band_projectability(vasprun, projections, num_excluded)
        summary = projectability_summary(
            energies, projectabilities, dis_froz_max, dis_win_max
        )
        print(
            f"  projectability: {summary['frozen_median']:.2f} frozen | "
            f"{summary['window_median']:.2f} between the windows"
        )
        if summary["low"]:
            print(
                f"  {summary['low']} of {summary['window_states']} states between the windows "
                f"are below {summary['threshold']:.2f}, i.e. barely atomic"
            )
        if not args.no_plot:
            plot_path = plot_projectability(
                energies,
                projectabilities,
                summary,
                vasprun.efermi,
                dis_froz_max,
                dis_win_max,
                output_path=output_dir / "projectability.png",
                title=f"{args.mp_id}: band projectability onto the Wannier manifold",
            )
            print(f"Wrote projectability plot to {plot_path}")
    except Exception as error:
        print(f"Could not compute the band projectability ({error}); the SCF run needs LORBIT=11 for it")

    if not args.no_plot:
        try:
            plot_path = plot_dos_windows(
                vasprun,
                projections,
                dis_froz_max,
                dis_win_max,
                output_path=output_dir / "dos_windows.png",
                title=f"{args.mp_id}: projected DOS and disentanglement windows",
            )
            print(f"Wrote window check plot to {plot_path}")
        except Exception as error:
            print(f"Could not write the window check plot ({error}); the SCF run needs LORBIT=11 for it")

    if not args.no_sbatch:
        script_path = write_sbatch_script(output_dir, args)
        print(f"Wrote sbatch script to {script_path}")


if __name__ == "__main__":
    main()
