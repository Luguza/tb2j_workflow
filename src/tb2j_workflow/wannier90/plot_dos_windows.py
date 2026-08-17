"""Plot the SCF density of states together with the disentanglement windows.

The windows written into the .win block are derived from the eigenvalues
without any human looking at the band structure, so this figure is the check
on that: one panel per atomic species, the orbitals that species contributes
to the Wannier manifold filled in, and the windows drawn as vertical lines.

Reading it the way the Wannier90 tutorials do: everything filled below
``dis_froz_max`` is reproduced exactly by the Wannier functions, the states
between ``dis_froz_max`` and ``dis_win_max`` are the ones disentanglement gets
to choose from, and anything above ``dis_win_max`` is discarded. If a filled
orbital that matters for the magnetism (the transition-metal d states, say)
sticks out above ``dis_win_max``, or if the frozen window cuts through the
middle of the valence manifold, the automatic settings need a nudge.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun

# One colour per angular momentum channel, shared by every panel so that the
# same orbital reads the same way across species.
ORBITAL_COLORS = {"s": "tab:blue", "p": "tab:green", "d": "tab:red", "f": "tab:purple"}


def projected_orbitals(projections: list[str]) -> dict[str, list[str]]:
    """``["Fe:s;p;d", "O:s;p"] -> {"Fe": ["s", "p", "d"], "O": ["s", "p"]}``."""
    orbitals = {}
    for line in projections:
        element, _, listed = line.partition(":")
        orbitals[element] = listed.split(";")
    return orbitals


def view_bottom(energies, total_dos, efermi: float, floor: float = 30.0) -> float:
    """Energy the plot should start at, in eV.

    The bottom of the lowest occupied band, so that shallow semicore states
    such as O 2s stay visible, but never more than ``floor`` below E_F: deep
    semicore levels (Fe 3p sits ~50 eV down) are inside the frozen window
    whatever the settings are, and plotting them squashes the region the
    windows actually cut through into a few pixels.
    """
    occupied = energies[(total_dos > 1e-3 * total_dos.max()) & (energies > efermi - floor)]
    return float(occupied.min()) - 1.0 if occupied.size else efermi - floor


def plot_dos_windows(
    vasprun: Vasprun,
    projections: list[str],
    dis_froz_max: float,
    dis_win_max: float,
    output_path: Path,
    title: str = "",
) -> Path:
    """Write the per-species DOS figure. Raises if the SCF ran without LORBIT."""
    complete_dos = vasprun.complete_dos  # needs LORBIT >= 10 in the SCF INCAR
    energies = complete_dos.energies
    efermi = complete_dos.efermi
    highlighted = projected_orbitals(projections)

    energy_min = view_bottom(energies, sum(complete_dos.densities.values()), efermi)
    energy_max = dis_win_max + 0.1 * (dis_win_max - energy_min)
    inside = (energies >= energy_min) & (energies <= energy_max)

    element_dos = complete_dos.get_element_dos()
    elements = sorted(element_dos, key=lambda element: element.symbol)
    spin_polarised = len(vasprun.eigenvalues) > 1

    # The total DOS sits behind every panel as a reference for where the
    # element contributes relative to everything else. It is several times
    # larger than any single element, so it is rescaled to each panel rather
    # than flattening the curve the panel is actually about.
    total_densities = complete_dos.densities
    total_peak = max(float(densities[inside].max()) for densities in total_densities.values())

    figure, axes = plt.subplots(
        len(elements),
        1,
        figsize=(9.0, 2.3 * len(elements) + 1.0),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for axis, element in zip(axes, elements, strict=True):
        symbol = element.symbol
        spd = complete_dos.get_element_spd_dos(element)

        peak = max(float(densities[inside].max()) for densities in element_dos[element].densities.values())

        scale = peak / total_peak if total_peak > 0.0 else 0.0
        for spin, densities in total_densities.items():
            sign = 1.0 if spin is Spin.up else -1.0
            axis.fill_between(
                energies[inside],
                0.0,
                sign * scale * densities[inside],
                color="0.85",
                linewidth=0.0,
                zorder=0,
            )

        for orbital_type, dos in spd.items():
            orbital = orbital_type.name
            if orbital not in highlighted.get(symbol, []):
                continue
            for spin, densities in dos.densities.items():
                sign = 1.0 if spin is Spin.up else -1.0
                axis.fill_between(
                    energies[inside],
                    0.0,
                    sign * densities[inside],
                    color=ORBITAL_COLORS[orbital],
                    alpha=0.55,
                    linewidth=0.0,
                    label=f"{symbol} {orbital}" if spin is Spin.up else None,
                )

        for spin, densities in element_dos[element].densities.items():
            sign = 1.0 if spin is Spin.up else -1.0
            axis.plot(energies[inside], sign * densities[inside], color="0.3", linewidth=0.8)

        axis.set_ylim(-1.1 * peak if spin_polarised else 0.0, 1.1 * peak)
        axis.set_ylabel(f"{symbol} DOS\n(states/eV)")
        axis.legend(loc="upper right", fontsize="small", frameon=False)
        if spin_polarised:
            axis.axhline(0.0, color="0.3", linewidth=0.8)

        # The frozen window has no lower bound, so everything left of the red
        # line is reproduced exactly; the shading makes that immediate.
        axis.axvspan(energy_min, dis_froz_max, color="tab:red", alpha=0.05, linewidth=0.0)
        axis.axvline(efermi, color="black", linestyle=":", linewidth=1.2)
        axis.axvline(dis_froz_max, color="tab:red", linestyle="--", linewidth=1.2)
        axis.axvline(dis_win_max, color="tab:blue", linestyle="-.", linewidth=1.2)

    axes[-1].set_xlim(energy_min, energy_max)
    axes[-1].set_xlabel("E (eV)")

    window_handles = [
        plt.Rectangle((0, 0), 1, 1, color="0.85", label="total DOS (scaled per panel)"),
        plt.Line2D([], [], color="black", linestyle=":", label=f"$E_F$ = {efermi:.2f} eV"),
        plt.Line2D([], [], color="tab:red", linestyle="--", label=f"dis_froz_max = {dis_froz_max:.2f} eV"),
        plt.Line2D([], [], color="tab:blue", linestyle="-.", label=f"dis_win_max = {dis_win_max:.2f} eV"),
    ]
    figure.legend(
        handles=window_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize="small",
    )
    if title:
        figure.suptitle(title, y=0.995, fontsize="medium")

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
