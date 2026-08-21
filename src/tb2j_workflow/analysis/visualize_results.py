"""Plot the results of a finished TB2J + Vampire run.

Last step of the workflow: read what the two result-producing stages left on
disk and turn it into the four figures the run is actually judged on. It reads
files, never a database, so it works on any stage directory and re-running it
is free.

Two independent sources feed it, and each figure names which one it came from:

* ``work/<mp-id>/vampire/output``, the temperature sweep, plus ``vampire.mat``
  next to it for the per-sublattice saturation moments the magnetisation
  columns are normalised against. This gives ``magnetization.png``,
  ``susceptibility.png`` and ``inverse_susceptibility.png``.
* ``work/<mp-id>/tb2j/TB2J_results/TB2J.pickle``, the exchange parameters
  themselves, read back through TB2J's own ``SpinIO``. This gives
  ``exchange_graph.png`` -- the couplings *before* Vampire integrates them, so
  a transition temperature that looks wrong can be traced to the J it came
  from.

The ordering temperature is read off the peak of the susceptibility, which is
what ``gen-vampire`` requests ``output:mean-susceptibility`` for. M(T) only
bends over near the transition and where exactly it "reaches zero" depends on
the noise floor, whereas chi diverges at the transition, so its maximum is a
much sharper estimator. A sweep written before that flag existed has no chi
columns; the sweep reader detects that from the column count, the two
susceptibility figures are skipped, and the ordering temperature falls back to
the steepest point of the sublattice magnetisation -- which is biased low and
resolution-limited, so it is reported as such.
"""

from __future__ import annotations

import argparse
import math
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

SERIES_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Curie-Weiss fit window (see ``curie_weiss_fit``): the lowest temperature the
# window may reach, in units of T_c; the top fraction of the allowed range taken
# as certainly linear; how many standard errors the leading edge of the window
# may drift off the line before it stops widening; how many candidates in a row
# must fail that test to stop the widening; and the number of points the edge
# test averages over, which is also the shortest window worth fitting.
CW_TC_FLOOR = 1.5
CW_ANCHOR_FRAC = 0.25
CW_NOISE_TOL = 2.0
CW_STOP_RUN = 3
CW_MIN_POINTS = 8


class ResultsError(RuntimeError):
    """The stage directories cannot be read as a finished workflow."""


# ---------------------------------------------------------------------------
# Reading what the stages wrote
# ---------------------------------------------------------------------------


@dataclass
class Material:
    """One sublattice of ``vampire.mat``, i.e. one magnetisation column."""

    index: int  # 1-based, matching material[N] and the column order
    name: str
    element: str
    moment: float  # atomic spin moment in muB, the M_sat of its column


@dataclass
class Sweep:
    """The temperature sweep of ``output``, split into named columns.

    ``material_m`` and ``total_m`` are magnetisation *lengths* normalised to
    saturation, so they run from 1 to 0. ``chi`` is ``None`` for a run made
    before ``gen-vampire`` requested the susceptibility.
    """

    temperature: np.ndarray
    material_m: list[np.ndarray]
    total_m: np.ndarray
    chi: dict[str, np.ndarray] | None

    @property
    def sublattice_m(self) -> np.ndarray:
        """Mean of the sublattice magnetisations.

        This, not ``total_m``, is the order parameter to look at: the moments
        all start along +z, so an antiferromagnet relaxes into its Neel state
        during equilibration and ``total_m`` is ~0 at every temperature.
        """
        return np.mean(self.material_m, axis=0)


def read_materials(mat_path: Path) -> list[Material]:
    """Parse ``vampire.mat`` into one ``Material`` per sublattice."""
    if not mat_path.is_file():
        raise ResultsError(
            f"{mat_path} is missing, so the Vampire stage was never written"
        )

    values: dict[str, str] = {}
    for line in mat_path.read_text().splitlines():
        line = line.split("#", 1)[0]
        key, sep, value = line.partition("=")
        if sep:
            values[key.replace(" ", "")] = value.strip()

    try:
        count = int(values["material:num-materials"])
    except (KeyError, ValueError) as error:
        raise ResultsError(
            f"{mat_path} has no readable 'material:num-materials': {error}"
        ) from error

    materials = []
    for index in range(1, count + 1):
        prefix = f"material[{index}]:"
        # The moment carries a "!muB" unit suffix; the number is the first field.
        raw_moment = values.get(f"{prefix}atomic-spin-moment", "0").split()[0]
        try:
            moment = float(raw_moment)
        except ValueError as error:
            raise ResultsError(
                f"{mat_path} has an unreadable moment for material {index}: {error}"
            ) from error
        materials.append(
            Material(
                index=index,
                name=values.get(f"{prefix}material-name", f"material {index}"),
                element=values.get(f"{prefix}material-element", ""),
                moment=moment,
            )
        )
    return materials


def read_sweep(output_path: Path, num_materials: int) -> Sweep:
    """Parse Vampire's ``output`` given how many magnetisation columns to expect.

    The file carries no column names -- Vampire writes one field per ``output:``
    line of the input, in that order -- so the layout is reconstructed from the
    material count: temperature, one magnetisation per material, the whole-system
    magnetisation, and then, if ``output:mean-susceptibility`` was requested, the
    four susceptibility columns chi_x, chi_y, chi_z, chi_m.
    """
    if not output_path.is_file():
        raise ResultsError(
            f"{output_path} does not exist: the Vampire run has not produced results yet "
            f"(submit {output_path.parent}/submit.sh)"
        )

    table = np.loadtxt(output_path, comments="#", ndmin=2)
    if table.size == 0:
        raise ResultsError(
            f"{output_path} has no data rows: the Vampire run wrote nothing"
        )

    without_chi = 2 + num_materials
    with_chi = without_chi + 4
    if table.shape[1] == with_chi:
        chi = {name: table[:, without_chi + i] for i, name in enumerate("xyzm")}
    elif table.shape[1] == without_chi:
        chi = None
    else:
        raise ResultsError(
            f"{output_path} has {table.shape[1]} columns, but {num_materials} materials imply "
            f"{without_chi} without susceptibility or {with_chi} with it. The output: lines of "
            f"{output_path.parent}/input do not match the ones gen-vampire writes."
        )

    return Sweep(
        temperature=table[:, 0],
        material_m=[table[:, 1 + i] for i in range(num_materials)],
        total_m=table[:, 1 + num_materials],
        chi=chi,
    )


@dataclass
class Coupling:
    """One drawn exchange bond between two magnetic sites.

    ``count`` is how many symmetry-equivalent bonds -- same pair, same distance,
    same J -- were collapsed into this one, which includes the (R, i, j) and
    (-R, j, i) pair that TB2J stores for every bond.
    """

    i: int  # spin index, i <= j
    j: int
    distance: float
    j_iso: float  # meV
    count: int
    shell: int  # 1-based neighbour shell of this pair, by distance


def read_couplings(results_dir: Path, j_min: float) -> tuple[object, list[Coupling]]:
    """Load TB2J's own results object and the couplings worth drawing.

    Every bond appears twice in ``exchange_Jdict``, once as (R, i, j) and once
    as (-R, j, i), so one of each pair is dropped before anything is counted.
    What is left is grouped on (site pair, distance, J): bonds that agree in all
    three are the same bond seen through a different lattice translation, and
    collapsing them is what keeps the figure from drawing one arc per periodic
    image. They are *not* averaged over a distance shell -- in a distorted cell
    two bonds of equal length can differ by more than a factor of two in J, and
    a shell average would hide exactly that.
    """
    try:
        from TB2J.io_exchange import SpinIO
    except ImportError as error:  # pragma: no cover - TB2J is a hard dependency
        raise ResultsError(
            f"TB2J is not importable, so the exchange graph cannot be drawn: {error}"
        ) from error

    if not (results_dir / "TB2J.pickle").is_file():
        raise ResultsError(
            f"{results_dir}/TB2J.pickle is missing, so the TB2J run did not finish"
        )
    spin_io = SpinIO.load_pickle(str(results_dir))

    # Collapse to one entry per (pair, distance, J), keeping a count.
    seen: set[tuple] = set()
    groups: dict[tuple[int, int, float, float], int] = defaultdict(int)
    for (cell, i, j), j_ev in spin_io.exchange_Jdict.items():
        j_iso = j_ev * 1000.0  # eV -> meV
        if abs(j_iso) < j_min:
            continue
        mate = (tuple(-np.asarray(cell)), j, i)
        if mate in seen:
            continue
        seen.add((tuple(cell), i, j))
        distance = spin_io.distance_dict[(cell, i, j)][1]
        groups[(min(i, j), max(i, j), round(distance, 2), round(j_iso, 3))] += 1

    # Shell index: rank the distinct distances of each pair, nearest first.
    shells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i, j, distance, _ in groups:
        if distance not in shells[(i, j)]:
            shells[(i, j)].append(distance)
    for distances in shells.values():
        distances.sort()

    couplings = [
        Coupling(
            i=i,
            j=j,
            distance=distance,
            j_iso=j_iso,
            count=count,
            shell=shells[(i, j)].index(distance) + 1,
        )
        for (i, j, distance, j_iso), count in groups.items()
    ]
    couplings.sort(key=lambda c: (c.i, c.j, c.distance, -abs(c.j_iso)))
    return spin_io, couplings


def site_labels(spin_io) -> list[str]:
    """Per-magnetic-site labels ``Fe1``, ``Fe2``, ... as TB2J numbers them.

    ``ind_atoms`` maps spin index to atom index; the number is the running count
    of that element in the structure, which is what ``exchange.out`` prints, so
    a coupling can be matched between the figure and the text output.
    """
    symbols = spin_io.atoms.get_chemical_symbols()
    numbering: dict[str, int] = defaultdict(int)
    labels_by_atom = {}
    for atom, symbol in enumerate(symbols):
        numbering[symbol] += 1
        labels_by_atom[atom] = f"{symbol}{numbering[symbol]}"
    return [labels_by_atom[spin_io.ind_atoms[spin]] for spin in range(spin_io.nspin)]


# ---------------------------------------------------------------------------
# Ordering temperature
# ---------------------------------------------------------------------------


def ordering_temperature(sweep: Sweep) -> tuple[float, str]:
    """Ordering temperature and the method it came from.

    The susceptibility peak is the estimator whenever chi was recorded: it
    diverges at the transition, so its maximum sits on it. Without chi the only
    thing left is the steepest descent of the sublattice magnetisation, which is
    biased below the transition -- M(T) starts falling before it -- and can be no
    finer than the temperature step, so callers label it as an estimate.
    """
    if sweep.chi is not None:
        return float(sweep.temperature[int(np.argmax(sweep.chi["m"]))]), "chi peak"

    order = np.argsort(sweep.temperature)
    temperature = sweep.temperature[order]
    magnetisation = sweep.sublattice_m[order]
    gradient = np.gradient(magnetisation, temperature)
    return float(temperature[int(np.argmin(gradient))]), "steepest dM/dT"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_magnetization(
    sweep: Sweep, materials: list[Material], title: str, outdir: Path
) -> None:
    """Plot M(T) per sublattice and for the whole system.

    Vampire reports magnetisation *lengths* normalised to each material's own
    saturation moment, so every curve starts at 1 regardless of how large the
    moment behind it is; the legend carries the M_sat each one was divided by.

    The whole-system curve is on the same axes because the contrast between it
    and the sublattices is what identifies the ordering: both finite below the
    transition is a ferromagnet, sublattices finite while the total sits at zero
    is an antiferromagnet. Its T=0 point is an artefact and is drawn as such --
    a collinear start in a collinear exchange field feels exactly zero torque, so
    nothing moves and the run reports its ferromagnetic starting state.
    """
    figure, axes = plt.subplots(figsize=(6, 4))

    for i, (material, column) in enumerate(
        zip(materials, sweep.material_m, strict=True)
    ):
        axes.plot(
            sweep.temperature,
            column,
            color=SERIES_COLORS[(i + 1) % len(SERIES_COLORS)],
            lw=2,
            alpha=0.7,
            label=f"{material.name} {material.index} ($M_s$ = {material.moment:.2f} $\\mu_B$)",
        )

    # Drop the T=0 row from the system curve only: it is the untorqued starting
    # state, and drawn it would put a vertical cliff at the origin of an
    # antiferromagnet that has nothing to do with a transition.
    finite = sweep.temperature > 0
    axes.plot(
        sweep.temperature[finite],
        sweep.total_m[finite],
        color=SERIES_COLORS[0],
        lw=2.5,
        label="whole system",
    )

    temperature, method = ordering_temperature(sweep)
    axes.axvline(
        temperature,
        color=INK_MUTED,
        ls="--",
        lw=1.5,
        label=f"$T_\\mathrm{{N/C}}$ = {temperature:.0f} K ({method})",
    )

    axes.set_xlabel("Temperature (K)")
    axes.set_ylabel(r"$M / M_\mathrm{sat}$")
    axes.set_title(f"{title} Vampire Monte Carlo")
    axes.grid(color=GRIDLINE, lw=0.5)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, fontsize=8)
    figure.tight_layout()

    out = outdir / "magnetization.png"
    figure.savefig(out, dpi=300)
    plt.close(figure)
    print(f"Wrote {out}")


def plot_susceptibility(sweep: Sweep, title: str, outdir: Path) -> None:
    """Plot the susceptibility curves; the peak of chi_m marks the transition.

    The mean chi_m is the bold curve and chi_x/chi_y/chi_z its spatial
    components. The ordering temperature is defined as the temperature of the
    chi_m maximum, so the marker line falls on the peak by construction.
    """
    if sweep.chi is None:
        print("Sweep has no susceptibility columns; skipping susceptibility plot.")
        return

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.plot(
        sweep.temperature, sweep.chi["m"], color=SERIES_COLORS[0], lw=2, label="mean"
    )
    for i, component in enumerate("xyz"):
        axes.plot(
            sweep.temperature,
            sweep.chi[component],
            color=SERIES_COLORS[(i + 1) % len(SERIES_COLORS)],
            lw=2,
            alpha=0.5,
            label=rf"$\chi_{component}$",
        )

    temperature, _ = ordering_temperature(sweep)
    axes.axvline(
        temperature,
        color=INK_MUTED,
        ls="--",
        lw=1.5,
        label=f"$T_\\mathrm{{N/C}}$ = {temperature:.0f} K",
    )

    axes.set_xlabel("Temperature (K)")
    axes.set_ylabel(r"Susceptibility $\chi$")
    axes.set_title(f"{title} Vampire Monte Carlo")
    axes.grid(color=GRIDLINE, lw=0.5)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    figure.tight_layout()

    out = outdir / "susceptibility.png"
    figure.savefig(out, dpi=300)
    plt.close(figure)
    print(f"Wrote {out}")


def curie_weiss_fit(t: np.ndarray, inv_chi: np.ndarray, tc: float) -> dict | None:
    """Fit 1/chi = a (T - theta) over the widest window that is still linear.

    The window always ends at the highest temperature sampled and grows
    downwards; the search never reaches below ``CW_TC_FLOOR * T_c``, because
    the critical region bends 1/chi over long before T_c itself and a window
    that dips into it drags the intercept towards the Monte Carlo T_c - which
    would make the mean-field estimate agree with it for the wrong reason.

    "Still linear" is judged against the run's own noise: the topmost
    ``CW_ANCHOR_FRAC`` of the allowed points is taken as certainly linear and
    its residual scatter is the reference sigma. The start is then walked down
    point by point, and each candidate is judged on the *mean* residual of the
    lowest ``CW_MIN_POINTS`` points in the window, accepted while that stays
    within ``CW_NOISE_TOL`` standard errors of zero. Testing the leading edge
    rather than the scatter of the whole window is what makes the cut sharp:
    a bend that has only reached the bottom few points is diluted to nothing in
    a residual RMS taken over a hundred, but it biases those points one way, so
    averaging them (noise falls as 1/sqrt(n), bias does not) sees it.

    It takes ``CW_STOP_RUN`` rejections in a row to stop the walk, and the
    window keeps the last accepted start, so it stays contiguous with the tail.
    Requiring a run is what separates noise from curvature: a lone point can
    scatter past the threshold anywhere in the linear region and would otherwise
    cut the window far above the real bend, while curvature only grows once it
    starts, so it fails every candidate from there down.

    Returns ``None`` when too few points sit above the floor to fit, otherwise
    ``{"t_start", "slope", "intercept", "theta", "n", "sigma"}`` with ``theta``
    the x-intercept, i.e. the mean-field ordering temperature.
    """
    order = np.argsort(t)
    t, inv_chi = t[order], inv_chi[order]

    def fit(start: int) -> tuple[float, float, float, float]:
        """Line over ``t[start:]``: slope, intercept, scatter, edge bias.

        The edge bias is the mean residual of the lowest ``CW_MIN_POINTS``
        points, i.e. the end of the window where curvature creeps in first.
        """
        slope, intercept = np.polyfit(t[start:], inv_chi[start:], 1)
        resid = inv_chi[start:] - (slope * t[start:] + intercept)
        dof = max(len(resid) - 2, 1)
        sigma = float(np.sqrt(np.sum(resid**2) / dof))
        return slope, intercept, sigma, float(np.mean(resid[:CW_MIN_POINTS]))

    (above,) = np.nonzero(t >= CW_TC_FLOOR * tc)
    if len(above) < 2 * CW_MIN_POINTS:
        return None
    first = int(above[0])

    # Anchor: the top slice of the allowed range, never shorter than the minimum
    # window, fitted on its own to measure the scatter of pure linear data.
    anchor = min(len(t) - round(CW_ANCHOR_FRAC * len(above)), len(t) - CW_MIN_POINTS)
    sigma_ref = fit(anchor)[2]
    max_bias = CW_NOISE_TOL * sigma_ref / np.sqrt(CW_MIN_POINTS)

    best = anchor
    failures = 0
    for start in range(anchor - 1, first - 1, -1):
        if abs(fit(start)[3]) > max_bias:
            failures += 1
            if failures >= CW_STOP_RUN:
                break
        else:
            failures = 0
            best = start

    slope, intercept, sigma, _ = fit(best)
    if slope <= 0:  # no paramagnetic tail to extrapolate along
        return None
    return {
        "t_start": float(t[best]),
        "slope": float(slope),
        "intercept": float(intercept),
        "theta": float(-intercept / slope),
        "n": len(t) - best,
        "sigma": sigma,
    }


def plot_inverse_susceptibility(sweep: Sweep, title: str, outdir: Path) -> None:
    """Plot 1/chi against temperature with the Curie-Weiss fit of its tail.

    Only the mean chi_m is inverted: the spatial components of
    ``plot_susceptibility`` are an order of magnitude noisier, and dividing by
    them turns that noise into spikes that would set the y-scale. Non-positive
    chi_m values (none in practice, but a Monte Carlo run can produce them) are
    dropped rather than inverted.

    Above the transition the curve approaches the Curie-Weiss straight line
    1/chi = a (T - theta) (see ``curie_weiss_fit`` for how the window is
    chosen). The fit needs a long paramagnetic tail, so it is the one figure
    that depends on how far past the transition the sweep was taken; when the
    sweep stops too soon it says so and draws the data alone.
    """
    if sweep.chi is None:
        print(
            "Sweep has no susceptibility columns; skipping inverse susceptibility plot."
        )
        return

    positive = sweep.chi["m"] > 0
    if not positive.any():
        print("No positive chi_m values; skipping inverse susceptibility plot.")
        return

    t = sweep.temperature[positive].astype(float)
    inv_chi = 1.0 / sweep.chi["m"][positive].astype(float)
    tc, _ = ordering_temperature(sweep)

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.plot(t, inv_chi, color=SERIES_COLORS[0], lw=2, label="mean")
    axes.axvline(
        tc, color=INK_MUTED, ls="--", lw=1.5, label=f"$T_\\mathrm{{N/C}}$ = {tc:.0f} K"
    )

    cw = curie_weiss_fit(t, inv_chi, tc)
    if cw is None:
        print(
            f"Too few points above {CW_TC_FLOOR:g} T_c = {CW_TC_FLOOR * tc:.0f} K for a Curie-Weiss "
            f"fit; re-run gen-vampire with a higher --temperature-max to get one."
        )
    else:
        # Extrapolate from the intercept up through the fitted window, and shade
        # the window itself so the extrapolated part is never mistaken for data.
        line_t = np.array([cw["theta"], t.max()])
        axes.plot(
            line_t,
            cw["slope"] * line_t + cw["intercept"],
            color=SERIES_COLORS[5],
            ls="--",
            lw=1.5,
            label=(
                rf"Curie-Weiss fit ($T \geq$ {cw['t_start']:.0f} K)"
                "\n"
                rf"$\theta$ = {cw['theta']:.0f} K"
            ),
        )
        axes.axvspan(cw["t_start"], t.max(), color=SERIES_COLORS[5], alpha=0.07, lw=0)
        axes.scatter(cw["theta"], 0.0, color=SERIES_COLORS[5], s=30, zorder=4)
        axes.set_ylim(bottom=0)
        axes.set_xlim(left=min(0.0, cw["theta"]) - 0.02 * t.max())
        print(
            f"Curie-Weiss fit over T >= {cw['t_start']:.0f} K ({cw['n']} points): "
            f"theta = {cw['theta']:.0f} K vs Monte Carlo T = {tc:.0f} K"
        )
        if cw["theta"] <= 0:
            print(
                "  theta < 0: the tail is an antiferromagnetic/ferrimagnetic hyperbola, so this is "
                "the asymptotic theta, not a mean-field ordering temperature."
            )

    axes.set_xlabel("Temperature (K)")
    axes.set_ylabel(r"Inverse susceptibility $1/\chi$")
    axes.set_title(f"{title} Vampire Monte Carlo")
    axes.grid(color=GRIDLINE, lw=0.5)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    figure.tight_layout()

    out = outdir / "inverse_susceptibility.png"
    figure.savefig(out, dpi=300)
    plt.close(figure)
    print(f"Wrote {out}")


def plot_exchange_graph(
    spin_io, couplings: list[Coupling], j_min: float, title: str, outdir: Path
) -> None:
    """Draw the interaction graph: magnetic sites joined by their J couplings.

    Nodes are the magnetic sites TB2J solved for, laid out on a circle and
    labelled with the site name and the moment ``exchange.out`` reports for it.
    Unlike a ground-state ordering diagram they carry no spin direction: the
    magnetic force theorem takes derivatives around one reference state in which
    every moment points the same way, so the sign of each *coupling* is the only
    magnetic information in the model, and it lives on the edges.

    Edges are the couplings above ``j_min``, each labelled with its own J_iso in
    meV under TB2J's convention E = -sum_ij J_ij S_i . S_j with unit spin
    vectors, so J < 0 is antiferromagnetic. Couplings between the same pair of
    sites are fanned out around the chord joining them and coloured by neighbour
    shell -- couplings of one pair that share a distance share a colour, and the
    legend gives that distance and how many bonds it covers. Couplings of a site
    to its own periodic images are self-loops.

    The cutoff is what makes this readable at all: TB2J tabulates a coupling for
    every site pair inside the Wannier supercell, thousands of which are numerical
    noise several angstrom out. The count of what was dropped is printed rather
    than hidden, because a material whose couplings do not fall off is telling
    you the Wannier Hamiltonian, not the magnetism.
    """
    if not couplings:
        print(f"No couplings above {j_min} meV; skipping exchange graph.")
        return

    labels = site_labels(spin_io)
    magmoms = np.asarray(spin_io.magmoms)
    sites = sorted(
        {index for coupling in couplings for index in (coupling.i, coupling.j)}
    )

    # Circular layout, first site at the top so the figure is reproducible.
    positions = {
        site: np.array(
            [
                math.cos(math.pi / 2 + 2 * math.pi * k / len(sites)),
                math.sin(math.pi / 2 + 2 * math.pi * k / len(sites)),
            ]
        )
        for k, site in enumerate(sites)
    }
    if len(sites) == 1:
        positions = {sites[0]: np.zeros(2)}
    centroid = np.mean(list(positions.values()), axis=0)

    # Colour encodes the neighbour shell, not the site pair: which pair an edge
    # belongs to is already given by the two nodes it runs between, so spending
    # colour on that would be redundant, and a pair-keyed palette wraps and
    # starts repeating itself as soon as a material has more than a few sites.
    # Shell indices are few by construction, so this legend never reuses a colour
    # for two different things.
    shell_keys = sorted({(c.i, c.j, c.shell) for c in couplings})
    shell_colors = {
        key: SERIES_COLORS[(key[2] - 1) % len(SERIES_COLORS)] for key in shell_keys
    }
    shell_distance = {(c.i, c.j, c.shell): c.distance for c in couplings}
    shell_count = defaultdict(int)
    for coupling in couplings:
        shell_count[(coupling.i, coupling.j, coupling.shell)] += coupling.count

    figure, axes = plt.subplots(figsize=(7.5, 6))

    def edge_label(xy: np.ndarray, j_iso: float, color: str) -> None:
        axes.annotate(
            f"{j_iso:+.1f}",
            xy,
            color=color,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": "white",
                "ec": "none",
                "alpha": 0.85,
            },
        )

    # Inter-site edges: every one is bowed by the same base radius so that a
    # chord through the middle of the ring does not run over its own label, and
    # the couplings of one pair fan out around that base. The fan spacing shrinks
    # as the pair gains couplings, so a pair with six of them still fits inside
    # the figure instead of looping out past the ring.
    base_rad = 0.16
    inter = [c for c in couplings if c.i != c.j]
    for pair in sorted({(c.i, c.j) for c in inter}):
        group = [c for c in inter if (c.i, c.j) == pair]
        spacing = min(0.34, 1.25 / len(group))
        for k, coupling in enumerate(group):
            rad = base_rad + spacing * (k - (len(group) - 1) / 2)
            color = shell_colors[(coupling.i, coupling.j, coupling.shell)]
            start, end = positions[pair[0]], positions[pair[1]]
            axes.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-",
                    color=color,
                    lw=2.2,
                    shrinkA=28,
                    shrinkB=28,
                    zorder=2,
                )
            )
            # Midpoint of the arc3 quadratic Bezier: the chord midpoint pushed
            # perpendicular by half the control-point offset matplotlib uses.
            delta = end - start
            mid = (start + end) / 2 + 0.5 * rad * np.array([delta[1], -delta[0]])
            edge_label(mid, coupling.j_iso, color)

    # Self-couplings: a site to its own periodic image, drawn as a loop pushed
    # radially outwards so it never overlaps the ring.
    loops = [c for c in couplings if c.i == c.j]
    for site in sorted({c.i for c in loops}):
        group = [c for c in loops if c.i == site]
        direction = positions[site] - centroid
        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        for k, coupling in enumerate(group):
            radius = 0.20 + 0.13 * k
            centre = positions[site] + direction * radius
            color = shell_colors[(coupling.i, coupling.j, coupling.shell)]
            axes.add_patch(
                plt.Circle(centre, radius, fill=False, lw=2.2, color=color, zorder=2)
            )
            edge_label(centre + direction * radius, coupling.j_iso, color)

    for site in sites:
        axes.scatter(
            *positions[site],
            s=3000,
            zorder=3,
            color=SERIES_COLORS[4],
            edgecolors="white",
            linewidths=2,
        )
        axes.annotate(
            f"{labels[site]}\n" rf"${magmoms[spin_io.ind_atoms[site]]:.2f}\,\mu_B$",
            positions[site],
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=4,
            linespacing=1.4,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=shell_colors[key],
            lw=2.5,
            label=(
                f"{labels[key[0]]}-{labels[key[1]]} shell {key[2]}:  "
                f"$d$ = {shell_distance[key]:.2f} $\\AA$, {shell_count[key]} "
                f"bond{'' if shell_count[key] == 1 else 's'}"
            ),
        )
        for key in shell_keys
    ]
    columns = 2 if len(handles) > 8 else 1
    axes.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=columns,
        fontsize=8,
        handlelength=1.6,
        borderaxespad=0.0,
        title=(
            r"edge labels: $J_\mathrm{iso}$ in meV, $E = -\sum_{ij} J_{ij}\,"
            r"\mathbf{S}_i\!\cdot\!\mathbf{S}_j$ ($J<0$ antiferromagnetic)"
        ),
        title_fontsize=8,
    )

    axes.set_title(f"{title} exchange couplings, $|J| \\geq {j_min:g}$ meV", pad=2)
    axes.set_aspect("equal")
    axes.margins(0.22)
    axes.axis("off")
    figure.tight_layout()

    out = outdir / "exchange_graph.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mp_id", help="Materials Project ID, e.g. mp-19770")
    parser.add_argument(
        "--vampire-dir",
        type=Path,
        default=None,
        help="Directory holding the finished Vampire run (default: ./work/<mp_id>/vampire)",
    )
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
        help="Directory to write the figures to (default: ./work/<mp_id>/results)",
    )
    parser.add_argument(
        "--j-min",
        type=float,
        default=0.5,
        help="Smallest |J_iso| in meV drawn in the exchange graph; raise it for a less crowded "
        "figure (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = cli()
    vampire_dir = args.vampire_dir or Path(f"./work/{args.mp_id}/vampire")
    tb2j_dir = args.tb2j_dir or Path(f"./work/{args.mp_id}/tb2j")
    outdir = args.output_dir or Path(f"./work/{args.mp_id}/results")
    outdir.mkdir(parents=True, exist_ok=True)

    materials = read_materials(vampire_dir / "vampire.mat")
    sweep = read_sweep(vampire_dir / "output", len(materials))
    spin_io, couplings = read_couplings(tb2j_dir / "TB2J_results", args.j_min)

    formula = spin_io.atoms.get_chemical_formula()
    title = f"{args.mp_id} ({formula})"

    temperature, method = ordering_temperature(sweep)
    total = len(spin_io.exchange_Jdict)
    drawn = sum(coupling.count for coupling in couplings)
    print(
        f"{title}: {len(materials)} magnetic sublattices, sweep of {len(sweep.temperature)} temperatures"
    )
    print(f"  ordering temperature: {temperature:.0f} K (from the {method})")
    print(
        f"  couplings:            {drawn} of {total} tabulated are >= {args.j_min} meV, "
        f"drawn as {len(couplings)} edges"
    )

    plot_magnetization(sweep, materials, title, outdir)
    plot_susceptibility(sweep, title, outdir)
    plot_inverse_susceptibility(sweep, title, outdir)
    plot_exchange_graph(spin_io, couplings, args.j_min, title, outdir)


if __name__ == "__main__":
    main()
