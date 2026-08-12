"""Plot how much atomic character the SCF bands carry, against the windows.

The disentanglement windows are chosen by counting bands: ``dis_win_max`` is
pushed up until every k-point has at least num_wann states below it, without
ever asking whether those states look anything like the orbitals being
projected onto. This figure asks that second question. For every band and
k-point it sums VASP's site-projected weights over the orbitals that make up
the Wannier manifold, which is the projectability that drives the SCDM
protocol of aiida-wannier90-workflows (Vitale et al., npj Comput. Mater. 6, 66
(2020)) -- computed here as a diagnostic rather than used to pick parameters.

Read it relatively, not absolutely. VASP only counts the weight that falls
inside the PAW spheres, so even a perfectly atomic band lands around 0.8 rather
than 1.0, and the number is not comparable to the pseudo-atomic projectabilities
the SCDM papers quote. What carries meaning is the *drop* from the frozen
region, where the bands are reproduced exactly, to the states between the two
windows, which is all disentanglement gets to choose from. States there that
sit far below the frozen level are free-electron-like: nothing in the manifold
resembles them, and the Wannier functions that come out of mixing them in are
correspondingly less localised.

A large drop is a statement about the upper edge of the manifold being
entangled with the free-electron continuum, not necessarily a problem to fix.
TB2J samples the occupied states, which are all deep inside the frozen region,
so the exchange integrals usually survive it -- but it is worth knowing which
materials it happens in.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.vasp.outputs import Vasprun

from .plot_dos_windows import projected_orbitals

# A state carrying less than this fraction of the frozen-region projectability
# has little enough atomic character to be worth counting in the summary.
LOW_PROJECTABILITY_FRACTION = 0.5

# Never plot deeper than this below E_F: the deep semicore levels are inside the
# frozen window whatever the settings are, and plotting them squashes the region
# the windows cut through into a few pixels. Matches plot_dos_windows.
VIEW_FLOOR = 30.0

SPIN_STYLES = {
    Spin.up: ("tab:blue", "o", "spin up"),
    Spin.down: ("tab:orange", "v", "spin down"),
}


def manifold_mask(structure: Structure, projections: list[str], num_orbitals: int) -> np.ndarray:
    """``[natoms, norbitals]`` mask of the projections spanning the manifold.

    The orbital axis of VASP's projections is lm-resolved in the order of
    pymatgen's ``Orbital`` enum (s, py, pz, px, dxy, ...), so LORBIT has to have
    been 11: the l-resolved output of LORBIT=10 has a shorter axis that would
    silently map onto the wrong orbitals.
    """
    if num_orbitals not in (9, 16):
        raise ValueError(
            f"the site projections have {num_orbitals} orbitals per atom rather than 9 or 16; "
            "the SCF run needs LORBIT = 11 for lm-resolved projections"
        )

    orbitals = projected_orbitals(projections)
    mask = np.zeros((len(structure), num_orbitals), dtype=bool)
    for atom, site in enumerate(structure):
        wanted = orbitals.get(site.specie.symbol, [])
        for index in range(num_orbitals):
            mask[atom, index] = Orbital(index).orbital_type.name in wanted
    return mask


def band_projectability(
    vasprun: Vasprun, projections: list[str], num_excluded: int = 0
) -> tuple[dict[Spin, np.ndarray], dict[Spin, np.ndarray]]:
    """Band energies and their projectability onto the manifold.

    Both are ``{spin: [nkpoints, nbands]}``. The bands excluded by index from
    the wannierisation are dropped, since they are not states disentanglement
    can pick and counting them would flatter the frozen-region average.
    """
    projected = vasprun.projected_eigenvalues  # needs LORBIT = 11 in the SCF INCAR
    if not projected:
        raise ValueError("the SCF run wrote no site-projected eigenvalues; it needs LORBIT = 11")

    mask = manifold_mask(
        vasprun.final_structure, projections, next(iter(projected.values())).shape[3]
    )
    energies = {
        spin: eigenvalues[:, num_excluded:, 0] for spin, eigenvalues in vasprun.eigenvalues.items()
    }
    projectabilities = {
        spin: weights[:, num_excluded:][..., mask].sum(axis=-1) for spin, weights in projected.items()
    }
    return energies, projectabilities


def projectability_summary(
    energies: dict[Spin, np.ndarray],
    projectabilities: dict[Spin, np.ndarray],
    dis_froz_max: float,
    dis_win_max: float,
) -> dict[str, float]:
    """Median projectability either side of ``dis_froz_max``, and the tail below it.

    ``low`` counts the states disentanglement can choose from that carry less
    than ``LOW_PROJECTABILITY_FRACTION`` of the frozen-region median, i.e. the
    ones with no real counterpart in the projections.
    """
    flat_energies = np.concatenate([array.ravel() for array in energies.values()])
    flat_projectabilities = np.concatenate([array.ravel() for array in projectabilities.values()])

    frozen = flat_projectabilities[flat_energies <= dis_froz_max]
    window = flat_projectabilities[
        (flat_energies > dis_froz_max) & (flat_energies <= dis_win_max)
    ]

    frozen_median = float(np.median(frozen)) if frozen.size else float("nan")
    threshold = LOW_PROJECTABILITY_FRACTION * frozen_median
    return {
        "frozen_median": frozen_median,
        "window_median": float(np.median(window)) if window.size else float("nan"),
        "threshold": threshold,
        "low": int((window < threshold).sum()),
        "window_states": int(window.size),
    }


def plot_projectability(
    energies: dict[Spin, np.ndarray],
    projectabilities: dict[Spin, np.ndarray],
    summary: dict[str, float],
    efermi: float,
    dis_froz_max: float,
    dis_win_max: float,
    output_path: Path,
    title: str = "",
) -> Path:
    """Write the projectability scatter figure."""
    # Start at the lowest band that is not deep semicore, so that the figure
    # covers the same energies as the DOS one and can be read next to it.
    flat = np.concatenate([array.ravel() for array in energies.values()])
    shallow = flat[flat > efermi - VIEW_FLOOR]
    energy_min = float(shallow.min()) - 1.0 if shallow.size else efermi - VIEW_FLOOR
    energy_max = dis_win_max + 0.1 * (dis_win_max - energy_min)

    figure, axis = plt.subplots(figsize=(9.0, 4.0))

    # Same shading as the DOS figure: everything left of the red line is
    # reproduced exactly, the band between the lines is what gets chosen from.
    axis.axvspan(energy_min, dis_froz_max, color="tab:red", alpha=0.05, linewidth=0.0)
    axis.axvspan(dis_froz_max, dis_win_max, color="tab:blue", alpha=0.05, linewidth=0.0)

    for spin, values in projectabilities.items():
        color, marker, _ = SPIN_STYLES[spin]
        axis.scatter(
            energies[spin].ravel(),
            values.ravel(),
            s=6.0,
            color=color,
            marker=marker,
            alpha=0.35,
            linewidths=0.0,
        )

    # The frozen level is the reference the rest of the plot is read against,
    # since the PAW-sphere weights never reach 1.
    axis.axhline(summary["frozen_median"], color="0.3", linewidth=0.9, linestyle="-")
    axis.axhline(summary["threshold"], color="0.3", linewidth=0.9, linestyle=":")
    axis.axvline(efermi, color="black", linestyle=":", linewidth=1.2)
    axis.axvline(dis_froz_max, color="tab:red", linestyle="--", linewidth=1.2)
    axis.axvline(dis_win_max, color="tab:blue", linestyle="-.", linewidth=1.2)

    axis.set_xlim(energy_min, energy_max)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("E (eV)")
    axis.set_ylabel("projectability onto\nthe Wannier manifold")

    handles = [
        plt.Line2D(
            [], [], color=SPIN_STYLES[spin][0], marker=SPIN_STYLES[spin][1], linestyle="",
            label=SPIN_STYLES[spin][2] if len(projectabilities) > 1 else "states",
        )
        for spin in projectabilities
    ] + [
        plt.Line2D([], [], color="0.3", linewidth=0.9, label=f"frozen median = {summary['frozen_median']:.2f}"),
        plt.Line2D([], [], color="0.3", linewidth=0.9, linestyle=":", label=f"{LOW_PROJECTABILITY_FRACTION:g}x frozen median"),
        plt.Line2D([], [], color="black", linestyle=":", label=f"$E_F$ = {efermi:.2f} eV"),
        plt.Line2D([], [], color="tab:red", linestyle="--", label=f"dis_froz_max = {dis_froz_max:.2f} eV"),
        plt.Line2D([], [], color="tab:blue", linestyle="-.", label=f"dis_win_max = {dis_win_max:.2f} eV"),
    ]
    axis.legend(handles=handles, loc="lower left", fontsize="small", frameon=False, ncol=2)
    if title:
        axis.set_title(title, fontsize="medium")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
