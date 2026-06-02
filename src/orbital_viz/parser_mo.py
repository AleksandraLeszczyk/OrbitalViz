from __future__ import annotations

import numpy as np


def read_molden_c_matrix(filepath: str) -> np.ndarray:
    """
    Read the AO→MO coefficient matrix from the ``[MO]`` section of a Molden file.

    Parameters
    ----------
    filepath : str
        Path to the ``.molden`` file.

    Returns
    -------
    np.ndarray, shape ``(N_AO, N_MO)``
        Column *n* holds the AO coefficients of the *n*-th molecular orbital.
    """
    mo_matrix: list[list[float]] = []
    current_mo: list[float] = []
    in_mo_section = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("[MO]"):
                in_mo_section = True
                continue

            if line.startswith("["):
                if in_mo_section:
                    if current_mo:
                        mo_matrix.append(current_mo)
                    break

            if not in_mo_section:
                continue

            # Metadata lines: "Sym= ...", "Ene= ...", "Alpha", "Occup= ..." etc.
            if "=" in line or line.isalpha():
                if current_mo:
                    mo_matrix.append(current_mo)
                    current_mo = []
                continue

            parts = line.split()
            if len(parts) == 2:
                try:
                    # Fortran-style exponent: 1.23D-04 → 1.23E-04
                    coeff = float(parts[1].replace("D", "E").replace("d", "e"))
                    current_mo.append(coeff)
                except ValueError:
                    continue

    if current_mo:
        mo_matrix.append(current_mo)

    # mo_matrix shape is (N_MO, N_AO); transpose to (N_AO, N_MO)
    return np.array(mo_matrix).T
