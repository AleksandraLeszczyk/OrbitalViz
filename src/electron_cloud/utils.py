from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Periodic-table metadata
# ══════════════════════════════════════════════════════════════════════════════

_SYMBOL: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    35: "Br",
    53: "I",
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

_COV_R: dict[int, float] = {  # covalent radius / Å  (Alvarez 2008)
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39,
}

_VDW_R: dict[int, float] = {  # van-der-Waals radius / Å  (display only)
    1: 0.53,
    6: 0.77,
    7: 0.75,
    8: 0.73,
    9: 0.71,
    15: 1.06,
    16: 1.02,
    17: 0.99,
    35: 1.14,
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