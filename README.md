# tb2j_workflow

TB2J exchange-parameter workflow: generate VASP SCF inputs for a Materials
Project structure, then feed the output into Wannier90 and .

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
