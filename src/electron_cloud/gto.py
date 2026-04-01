from __future__ import annotations

from typing import Any, List, Tuple, Self

import numpy as np

from electron_cloud.utils import _CART, _prim_norm, _SYMBOL


class BasisGTO:
    """
    Contracted Gaussian-Type Orbital (CGTO) basis set for a molecule.

    Each *shell* groups several primitive Gaussians that share the same
    exponents but are combined with different contraction coefficients into a
    single contracted function.  A p-shell (l = 1) automatically expands
    into three Cartesian functions (px, py, pz); a d-shell into six, etc.

    Parameters
    ----------
    atoms : list[int]
        Atomic numbers, one entry per atom
        (e.g. ``[8, 1, 1]`` for water).
    coordinates : list[list[float]]
        Cartesian atomic positions in **Ångström**, one ``[x, y, z]`` row
        per atom.
    number_of_primitives : list[int]
        Number of primitive Gaussian functions in *each* shell.
        Length equals the total number of shells.
    contraction : list[float]
        Flat list of contraction coefficients for every primitive in every
        shell, stored in shell-major / primitive-minor order.
    Alpha : list[float]
        Flat list of Gaussian exponents, in the same order as *contraction*.
    shell_types : list[int]
        Angular-momentum quantum number per shell
        (0 = s, 1 = p, 2 = d, 3 = f, …).
        Length equals the total number of shells.
    shell_to_atom : list[int]
        Zero-based atom index that each shell is centred on.
        Length equals the total number of shells.

    Attributes
    ----------
    atoms : np.ndarray[int],   shape (N_atoms,)
    coordinates : np.ndarray,  shape (N_atoms, 3)  [Å]
    n_atoms : int
    n_basis : int
        Total number of Cartesian AO functions (sum of 2l+1 or l+1 per shell
        in Cartesian convention).

    Examples
    --------
    >>> basis = BasisGTO(
    ...     atoms                = [1, 1],
    ...     coordinates          = [[0., 0., -0.370], [0., 0., 0.370]],
    ...     number_of_primitives = [3, 3],
    ...     contraction          = [0.1543, 0.5353, 0.4446] * 2,
    ...     Alpha                = [3.4253, 0.6239, 0.1689] * 2,
    ...     shell_types          = [0, 0],
    ...     shell_to_atom        = [0, 1],
    ... )
    >>> basis.n_basis
    2
    """

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_pybest_basis(
        cls,
        basis: Any,
    ) -> Self:

        return cls(
            atoms=basis.atom,
            coordinates=basis.coordinates,
            number_of_primitives=basis.nprim,
            contraction=basis.contraction,
            alpha=basis.alpha,
            shell_types=basis.shell_types,
            shell_to_atom=basis.shell2atom,
        )

    def __init__(
        self,
        atoms: List[int],
        coordinates: List[List[float]],
        number_of_primitives: List[int],
        contraction: List[float],
        alpha: List[float],
        shell_types: List[int],
        shell_to_atom: List[int],
    ):

        self.atoms = np.asarray(atoms, dtype=int)
        self.coordinates = np.asarray(coordinates, dtype=float)  # (N_at, 3) Å
        self.n_atoms = int(self.atoms.size)

        number_of_primitives = number_of_primitives
        shell_types = shell_types
        shell_to_atom = shell_to_atom

        a_arr = np.asarray(alpha, dtype=float)
        c_arr = np.asarray(contraction, dtype=float)

        if len(number_of_primitives) != len(shell_types):
            raise ValueError(
                "number_of_primitives and shell_types must have the same length "
                f"(got {len(number_of_primitives)} and {len(shell_types)})"
            )
        if len(shell_types) != len(shell_to_atom):
            raise ValueError(
                "shell_types and shell_to_atom must have the same length "
                f"(got {len(shell_types)} and {len(shell_to_atom)})"
            )

        # Internal per-AO storage (indexed by AO function index 0…n_basis-1)
        self._centers: list[np.ndarray] = []  # (3,)
        self._lxlylz: list[Tuple[int, int, int]] = []
        self._alphas: list[np.ndarray] = []  # (K,)  exponents
        self._nc: list[np.ndarray] = []  # (K,)  norm × coeff

        ptr = 0  # pointer into the flat Alpha / contraction arrays
        for n_prim, l, at in zip(number_of_primitives, shell_types, shell_to_atom):
            if at >= self.n_atoms:
                raise IndexError(
                    f"shell_to_atom contains index {at} but there are only "
                    f"{self.n_atoms} atom(s)."
                )
            center = self.coordinates[at]
            alphas = a_arr[ptr : ptr + n_prim]
            coeffs = c_arr[ptr : ptr + n_prim]
            ptr += n_prim

            for lx, ly, lz in _CART.get(l, [(0, 0, 0)]):
                norms = np.array([_prim_norm(a, lx, ly, lz) for a in alphas])
                self._centers.append(center.copy())
                self._lxlylz.append((lx, ly, lz))
                self._alphas.append(alphas.copy())
                self._nc.append(norms * coeffs)

        self.n_basis = len(self._centers)

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        atoms_str = ", ".join(_SYMBOL.get(int(z), str(z)) for z in self.atoms)
        return (
            f"BasisGTO(atoms=[{atoms_str}], "
            f"n_atoms={self.n_atoms}, n_basis={self.n_basis})"
        )

    # ── low-level evaluation ──────────────────────────────────────────────

    def _eval_ao(
        self,
        idx: int,
        xf: np.ndarray,  # shape (N,), ravelled grid
        yf: np.ndarray,
        zf: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a single contracted AO on flat coordinate arrays."""
        cx, cy, cz = self._centers[idx]
        lx, ly, lz = self._lxlylz[idx]
        dx, dy, dz = xf - cx, yf - cy, zf - cz
        r2 = dx * dx + dy * dy + dz * dz

        # Radial part: sum_k  (norm_k × coeff_k) × exp(−alpha_k × r²)
        # Shape trick: outer product → (K, N) then sum over K
        radial = np.dot(self._nc[idx], np.exp(np.outer(-self._alphas[idx], r2)))

        # Angular part: x^lx y^ly z^lz  (skipped for s-functions)
        l = lx + ly + lz
        if l == 0:
            return radial
        angular = np.ones_like(dx)
        if lx:
            angular = angular * dx**lx
        if ly:
            angular = angular * dy**ly
        if lz:
            angular = angular * dz**lz
        return angular * radial

    # ── public evaluation API ─────────────────────────────────────────────

    def evaluate_aos(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate all contracted AOs simultaneously on an arbitrary grid.

        Parameters
        ----------
        x, y, z : np.ndarray, any shape
            Grid coordinates in Ångström.

        Returns
        -------
        np.ndarray, shape ``(n_basis, *x.shape)``
            ``result[μ, ...]`` is the value of AO μ on the grid.
        """
        shape = x.shape
        xf, yf, zf = x.ravel(), y.ravel(), z.ravel()
        out = np.empty((self.n_basis, xf.size), dtype=float)
        for i in range(self.n_basis):
            out[i] = self._eval_ao(i, xf, yf, zf)
        return out.reshape((self.n_basis,) + shape)

    def evaluate_mo(
        self,
        mo_coeffs: np.ndarray,
        n: int,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the *n*-th molecular orbital on a grid.

        .. math::
            \\psi_n(\\mathbf{r}) = \\sum_{\\mu} C_{\\mu n}\\, \\phi_{\\mu}(\\mathbf{r})

        Parameters
        ----------
        mo_coeffs : np.ndarray, shape ``(n_basis, n_mo)``
            MO coefficient matrix  C.  Column *n* holds the AO coefficients
            of the *n*-th MO.
        n : int
            Zero-based MO index.
        x, y, z : np.ndarray, any (identical) shape
            Grid coordinates in Ångström.

        Returns
        -------
        np.ndarray, same shape as *x*
            ψₙ(r) on the grid.
        """
        if mo_coeffs.shape[0] != self.n_basis:
            raise ValueError(
                f"mo_coeffs has {mo_coeffs.shape[0]} rows but basis has "
                f"{self.n_basis} functions."
            )
        c = mo_coeffs[:, n]  # (n_basis,)
        aos = self.evaluate_aos(x, y, z)  # (n_basis, *shape)
        return np.einsum("i,i...->...", c, aos)  # contracts over AO index