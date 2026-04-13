from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Periodic-table metadata
# ══════════════════════════════════════════════════════════════════════════════

_SYMBOL: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba",
    # Lanthanides
    57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy",
    67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    # Period 6 Transition/Main Group
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
    82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
    # Period 7
    87: "Fr", 88: "Ra",
    # Actinides
    89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf",
    99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr"
}

# CPK colours (hex)
_COLOR: dict[int, str] = {
    1: "#FFFFFF",
    2: "#D9FFFF",
    3: "#CC80FF",
    4: "#C2FF00",
    5: "#FFB5B5",
    6: "#909090",
    7: "#3050F8",
    8: "#FF0D0D",
    9: "#90E050",
    10: "#B3E3F5",
    11: "#AB5CF2",
    12: "#8AFF00",
    13: "#BFA6A6",
    14: "#F0C8A0",
    15: "#FF8000",
    16: "#FFFF30",
    17: "#1FF01F",
    18: "#80D1E3",
    35: "#A62929",
    53: "#940094",
}

_COV_R: dict[int, float] = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
    31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75,
    41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 
    # Lanthanides
    57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96, 65: 1.94, 66: 1.92,
    67: 1.91, 68: 1.89, 69: 1.90, 70: 1.87, 71: 1.87,
    # Post-Lanthanides
    72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45,
    82: 1.46, 83: 1.48, 84: 1.40, 85: 1.50, 86: 1.50, 87: 2.60, 88: 2.21, 
    # Actinides
    89: 2.15, 90: 2.06, 91: 2.00, 92: 1.96, 93: 1.90, 94: 1.87, 95: 1.80, 96: 1.69, 97: 1.60, 98: 1.60,
    99: 1.60, 100: 1.60, 101: 1.60, 102: 1.60, 103: 1.60
}

_VDW_R: dict[int, float] = {
    # Period 1
    1: 1.20,   # H
    2: 1.40,   # He
    # Period 2
    3: 1.82,   # Li
    4: 1.53,   # Be
    5: 1.92,   # B
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    10: 1.54,  # Ne
    # Period 3
    11: 2.27,  # Na
    12: 1.73,  # Mg
    13: 1.84,  # Al
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    18: 1.88,  # Ar
    # Period 4
    19: 2.75,  # K
    20: 2.31,  # Ca
    21: 2.11,  # Sc
    22: 2.15,  # Ti
    23: 2.16,  # V
    24: 2.45,  # Cr
    25: 2.45,  # Mn
    26: 2.44,  # Fe
    27: 2.40,  # Co
    28: 1.63,  # Ni
    29: 1.40,  # Cu
    30: 1.39,  # Zn
    31: 1.87,  # Ga
    32: 2.11,  # Ge
    33: 1.85,  # As
    34: 1.90,  # Se
    35: 1.85,  # Br
    36: 2.02,  # Kr
    # Period 5
    37: 3.03,  # Rb
    38: 2.49,  # Sr
    39: 2.32,  # Y
    40: 2.23,  # Zr
    41: 2.18,  # Nb
    42: 2.17,  # Mo
    43: 2.16,  # Tc
    44: 2.13,  # Ru
    45: 2.10,  # Rh
    46: 1.63,  # Pd
    47: 1.72,  # Ag
    48: 1.58,  # Cd
    49: 1.93,  # In
    50: 2.17,  # Sn
    51: 2.06,  # Sb
    52: 2.06,  # Te
    53: 1.98,  # I
    54: 2.16,  # Xe
    # Period 6 (including Lanthanides)
    55: 3.43,  # Cs
    56: 2.68,  # Ba
    57: 2.43,  # La
    58: 2.42,  # Ce
    59: 2.40,  # Pr
    60: 2.39,  # Nd
    61: 2.38,  # Pm
    62: 2.36,  # Sm
    63: 2.35,  # Eu
    64: 2.34,  # Gd
    65: 2.33,  # Tb
    66: 2.31,  # Dy
    67: 2.30,  # Ho
    68: 2.29,  # Er
    69: 2.27,  # Tm
    70: 2.26,  # Yb
    71: 2.25,  # Lu
    72: 2.23,  # Hf
    73: 2.22,  # Ta
    74: 2.18,  # W
    75: 2.16,  # Re
    76: 2.16,  # Os
    77: 2.13,  # Ir
    78: 1.75,  # Pt
    79: 1.66,  # Au
    80: 1.55,  # Hg
    81: 1.96,  # Tl
    82: 2.02,  # Pb
    83: 2.07,  # Bi
    84: 1.97,  # Po
    85: 2.02,  # At
    86: 2.20,  # Rn
    # Period 7 (including Actinides)
    87: 3.48,  # Fr
    88: 2.83,  # Ra
    89: 2.47,  # Ac
    90: 2.45,  # Th
    91: 2.43,  # Pa
    92: 2.41,  # U
    93: 2.39,  # Np
    94: 2.37,  # Pu
    95: 2.35,  # Am
    96: 2.34,  # Cm
    97: 2.32,  # Bk
    98: 2.31,  # Cf
    99: 2.30,  # Es
    100: 2.29, # Fm
    101: 2.27, # Md
    102: 2.26, # No
    103: 2.25  # Lr
}

# Cartesian exponent triples (lx, ly, lz) — Gaussian / Molden ordering
_CART: dict[int, list[Tuple[int, int, int]]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (0, 2, 1),
        (1, 0, 2),
        (0, 1, 2),
        (1, 1, 1),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GTO mathematics
# ══════════════════════════════════════════════════════════════════════════════

def _dfact(n: int) -> int:
    """Double factorial  n!!,  with the convention  (−1)!! = 0!! = 1."""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def _prim_norm(alpha: float, lx: int, ly: int, lz: int) -> float:
    """
    Normalisation constant  N  such that

        ⟨φ|φ⟩ = 1   for   φ(r) = N · x^lx · y^ly · z^lz · exp(−α r²).

    Formula (Helgaker, Jørgensen, Olsen, eq. 6.6.21):

        N² = (2α/π)^(3/2)  ×  (4α)^l  /  [(2lx−1)!! (2ly−1)!! (2lz−1)!!]
    """
    l = lx + ly + lz
    N2 = (2.0 * alpha / math.pi) ** 1.5
    N2 *= (4.0 * alpha) ** l
    N2 /= _dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1)
    return math.sqrt(N2)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_bonds(
    coords: np.ndarray,
    atoms: np.ndarray,
    factor: float = 1.7,
) -> list[Tuple[int, int]]:
    """
    Infer bonds from interatomic distances vs. covalent-radii threshold.

    Two atoms *i* and *j* are bonded when
        d_{ij}  <  factor × (r_cov_i + r_cov_j).
    """
    bonds: list[Tuple[int, int]] = []
    for i in range(len(atoms)):
        ri = _COV_R.get(int(atoms[i]), 0.77)
        for j in range(i + 1, len(atoms)):
            rj = _COV_R.get(int(atoms[j]), 0.77)
            if np.linalg.norm(coords[i] - coords[j]) < factor * (ri + rj):
                bonds.append((i, j))
    return bonds


def _cylinder_mesh(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float = 0.08,
    n_sides: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a triangulated open cylinder between two 3-D points.

    Returns
    -------
    x, y, z : vertex coordinates
    i, j, k : triangle vertex indices  — ready for ``go.Mesh3d``.
    """
    axis = p2 - p1
    length = np.linalg.norm(axis)
    axis = axis / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Build a local orthonormal frame (u, v, axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    theta = np.linspace(0.0, 2.0 * math.pi, n_sides, endpoint=False)
    circle = radius * (
        np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    )  # (n, 3)

    verts = np.vstack([p1 + circle, p2 + circle])  # (2n, 3)

    # Triangulate the lateral surface
    ii, jj, kk = [], [], []
    for k in range(n_sides):
        kp = (k + 1) % n_sides
        ii += [k, k, n_sides + k]
        jj += [kp, n_sides + k, n_sides + kp]
        kk += [n_sides + k, kp, kp]

    return (
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        np.array(ii),
        np.array(jj),
        np.array(kk),
    )