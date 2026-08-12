# tb2j_workflow

TB2J exchange-parameter workflow: generate VASP SCF inputs for a Materials
Project structure, then feed the output into Wannier90 and .

## Wannierisation stage

`gen-wannier90 <mp-id>` reads a finished SCF run and writes a non-self-consistent
VASP run (`ICHARG=11` off the SCF `CHGCAR`, `ISYM=-1` for the full k-grid) with
`LWANNIER90=.TRUE.`, so VASP drives Wannier90 in-process and writes the
`*_hr.dat` / `*_centres.xyz` files that `wann2J.py` consumes.

This needs the VASP build that is linked against the Wannier90 library, which on
this cluster is the `Wan90` flavour of the `chem/vasp` module — the generated
`submit.sh` calls `vasp -f Wan90 -s std`. The plain `vasp` binary would silently
ignore `LWANNIER90`.

Nothing has to be chosen per material:

* **Projections** are every valence orbital in each element's POTCAR
  (`Fe_pv -> Fe:s;p;d`, `O -> O:s;p`), which fixes `num_wann`. When a POTCAR
  carries two shells of the same angular momentum (`Sr_sv` has 4s and 5s), the
  deeper one cannot be written in Wannier90's projection syntax, so those
  semicore bands are dropped via `exclude_bands` instead of being left to
  contaminate the window.
* **Disentanglement windows** are derived from the SCF eigenvalues rather than
  guessed. The NSCF run reuses the SCF density on the same mesh, so its
  eigenvalues are the SCF ones and Wannier90's two hard constraints can be
  satisfied exactly: `dis_froz_max` sits just below `min_k E_{num_wann+1}(k)`
  (never more than `num_wann` frozen states at any k) and `dis_win_max` above
  `max_k E_{num_wann}(k)` (never fewer than `num_wann` states in the window).

Since nothing is chosen by hand, the run also writes `dos_windows.png` next to
the inputs (`--no-plot` skips it): the SCF projected DOS, one panel per atomic
species with the orbitals that species contributes to the Wannier manifold
filled in, and `E_F`, `dis_froz_max` and `dis_win_max` drawn as vertical lines.
It is there to be looked at before submitting — states left of the red line are
reproduced exactly, those between red and blue are what disentanglement picks
from, and anything right of blue is thrown away. If the frozen window cuts
through the middle of the manifold that carries the magnetism, re-run with a
narrower `--projections` set or a different `--top-band-margin`. Drawing it
needs the SCF run to have `LORBIT=11` (pymatgen's `MPStaticSet` sets it); if it
is missing, the plot is skipped with a warning and the inputs are still written.

The windows are set by counting bands, which never asks whether the states they
admit resemble the orbitals being projected onto, so `projectability.png` and a
printed summary answer that second question. Every band and k-point is plotted
against the sum of its site-projected weights over the manifold orbitals — the
projectability that drives the SCDM protocol of aiida-wannier90-workflows, used
here to check the windows rather than to choose them. Read it relative to the
frozen level, not against 1.0: VASP only counts weight inside the PAW spheres,
so a fully atomic band lands near 0.9. What matters is the drop between the
frozen region and the states disentanglement chooses from; a large one means the
top of the manifold is entangled with the free-electron continuum. Since TB2J
samples the occupied states, well inside the frozen window, that is usually
tolerable — but worth knowing per material.

Because that derivation relies on the meshes matching, the KPOINTS file is
copied from the SCF run and there is deliberately no k-density flag; to
wannierise on a denser mesh, re-run the SCF stage at that density. If the SCF
did not have enough bands, the script fails with the `NBANDS` value to re-run
`gen-scf` with rather than emitting a silently degraded Hamiltonian.

## TB2J stage

`gen-tb2j <mp-id>` reads a finished wannierisation and writes the `wann2J.py`
run that turns the two spin channels of the real-space Wannier Hamiltonian into
Heisenberg exchange parameters via the magnetic force theorem. It writes no
inputs of its own — TB2J reads the Wannier files where they are, so the stage
directory holds only `submit.sh`, and `TB2J_results/` appears next to it.

Unlike the VASP stages this one is not a cluster module: `wann2J.py` is a
Python program from this project's virtualenv, so the generated script sources
`.venv/bin/activate` (the venv `gen-tb2j` itself ran from) to put it on `PATH`.
It also parallelises over the contour energy points with multiprocessing rather
than MPI, so the job asks for one task on one node with `--cpus-per-task`
cores and passes that same number as `--np`.

Everything `wann2J.py` needs comes out of the wannierisation directory:

* **Fermi level** from that run's `vasprun.xml`. It is the NSCF run whose
  eigenvalues the Wannier functions were fitted to, so its `E_F` is the one
  consistent with the Hamiltonian TB2J reads; the SCF value is the same number,
  since the NSCF run reuses the SCF density on the same mesh.
* **Magnetic elements** from the INCAR `MAGMOM`. The threshold defaults to
  1.0 μB rather than something small because pymatgen seeds *every* site with
  0.6 μB whether or not it is magnetic, and the magnetic transition metals with
  5 μB — at 0.5 μB the oxygens in an oxide come out "magnetic". Since this
  reads the starting guess and not the converged moments, check the printed
  element list, and override it with `--elements` for a material whose
  magnetism is not the one MAGMOM assumed.
* **k-mesh** from `KPOINTS`, i.e. the DFT mesh. A denser mesh is tempting but
  adds no information: the `_hr.dat` Hamiltonian only carries the real-space
  vectors inside the Wannier supercell that this mesh defines, so J(R) beyond
  it is computed from padding rather than data and shows up as long-range tails
  that look like physics. `--kmesh` overrides it for a convergence check.
* **File prefixes** `wannier90.up` / `wannier90.dn`, which is what VASP names
  the per-spin Wannier files of an `ISPIN=2` run — the same defaults TB2J uses.

The stage refuses to write a script for an `ISPIN=1` run (no spin splitting to
extract exchange from) and for a wannierisation that stopped before writing
`*_hr.dat` or `*_centres.xyz` for both spins, rather than leaving the failure
to be discovered in the queue.

## VASP POTCAR setup (pymatgen)

`scf/generate_scf_input.py` builds VASP input sets via pymatgen, which needs
`PMG_VASP_PSP_DIR` pointing at a directory of POTCAR files. This is
configured once per user account (not per-venv) via
`~/.config/.pmgrc.yaml`.

On this cluster, POTCARs are provided by the `chem/vasp` module at
`/opt/bwhpc/common/chem/vasp/<version>/pot/`, readable via ACL by users with
a VASP license. Setup, done once:

```bash
mkdir -p ~/.local/share/pmg_potcars
ln -s /opt/bwhpc/common/chem/vasp/6.5.1_ifxicx202421_avx2/pot/pot_pbe_paw_54/elements \
      ~/.local/share/pmg_potcars/POT_GGA_PAW_PBE_54
ln -s /opt/bwhpc/common/chem/vasp/6.5.1_ifxicx202421_avx2/pot/pot_pbe_paw_52/elements \
      ~/.local/share/pmg_potcars/POT_GGA_PAW_PBE_52

pmg config --add PMG_VASP_PSP_DIR ~/.local/share/pmg_potcars
```

This only symlinks into the module-provided POTCAR directory (no copying of
licensed VASP data) and registers the path in pymatgen's global settings
file. Since it lives in `$HOME`, it applies to every pymatgen install for
this user, across all venvs/projects — not just this repo.

Note: pymatgen will warn that `Fe_pv`/`O` POTCAR hashes match
`PBE_64`/`PBE_54_W_HASH`/`PBE_52_W_HASH` rather than plain `PBE_54`. The
POTCARs still work; this just means the cluster's "54" set is a
hash-stamped variant. Safe to ignore unless exact POTCAR provenance matters
for a given calculation.

Environment variables always override the yaml file, so any project that
sets `PMG_VASP_PSP_DIR` itself (e.g. `DFT_Pipeline2`, which points at its
own `.vasp_psp_compat` directory) is unaffected by this global default.
