# OrbitalViz

Visualise molecules and molecular orbitals (MOs) interactively in Python.

[<img width="870" height="212" alt="image" src="https://github.com/user-attachments/assets/2b398e11-624d-41cc-84e1-aab6d41798fd" />](https://aleksandraleszczyk.github.io/ElectronCloud/examples/ethylene_mo6_rhf.html)

> Click the image to open a live interactive example.

---

## Installation

```bash
pip install orbital_viz
```

Requires Python ≥ 3.11.

---

## Quick start

### From a Molden file

```python
from orbital_viz import parse_molden_to_dict, BasisGTO, read_molden_c_matrix, plot_molecular_orbital

# Parse geometry and basis set
data = parse_molden_to_dict("molecule.molden")
basis = BasisGTO(**data)

# Read MO coefficients
C = read_molden_c_matrix("molecule.molden", basis.n_basis)

# Plot MO index 5 (zero-based)
fig = plot_molecular_orbital(basis, C, n=5)
fig.show()
```

### From a custom basis (e.g. PyBEST output)

```python
from orbital_viz import BasisGTO, plot_molecular_orbital
import numpy as np

basis = BasisGTO(
    atoms=[6, 6],                         # Carbon, Carbon
    coordinates=[[-0.67, 0, 0], [0.67, 0, 0]],  # Å
    number_of_primitives=[3, 3],
    contraction=[0.154, 0.535, 0.444, 0.154, 0.535, 0.444],
    Alpha=[71.6, 13.0, 3.53, 71.6, 13.0, 3.53],
    shell_types=[0, 0],
    shell_to_atom=[0, 1],
)

C = np.load("mo_coefficients.npy")   # shape (n_basis, n_mo)
fig = plot_molecular_orbital(basis, C, n=0, isovalue=0.04, dark_bg=True)
fig.show()
```

---

## API reference

### `parse_molden_to_dict(filepath)`

Parses a standard Molden file and returns a dictionary ready to pass directly into `BasisGTO`.

```python
data = parse_molden_to_dict("molecule.molden")
# keys: atoms, coordinates, number_of_primitives,
#       contraction, Alpha, shell_types, shell_to_atom
```

---

### `read_molden_c_matrix(filepath, n_basis)`

Reads the MO coefficient matrix from a Molden file.

Returns an `np.ndarray` of shape `(n_basis, n_mo)`.

---

### `BasisGTO`

Contracted GTO basis set for a molecule. Handles Cartesian and spherical (solid-harmonic) AOs up to arbitrary angular momentum.

```python
BasisGTO(
    atoms,                  # list[int]  – atomic numbers
    coordinates,            # list[list[float]]  – XYZ positions in Å
    number_of_primitives,   # list[int]  – primitives per shell
    contraction,            # list[float]  – contraction coefficients (flat)
    Alpha,                  # list[float]  – exponents (flat)
    shell_types,            # list[int]  – angular momentum per shell (0=s,1=p,…)
    shell_to_atom,          # list[int]  – atom index per shell
)
```

Key attributes: `atoms`, `coordinates`, `n_atoms`, `n_basis`.

---

### `plot_molecular_orbital(Basis, mo_coeffs, n, **kwargs)`

Renders MO `n` as an interactive Plotly 3-D figure.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_points` | `50` | Grid resolution per axis — increase for smoother surfaces |
| `isovalue` | `0.05` | ψ value at which isosurfaces are drawn |
| `padding` | `2.0` | Extra space (Å) around the molecule bounding box |
| `opacity` | `0.45` | Isosurface opacity (0–1) |
| `atom_scale` | `0.25` | Scale factor on VDW radii for atom spheres |
| `bond_radius` | `0.08` | Bond cylinder radius (Å) |
| `title` | `None` | Figure title; defaults to `"MO #n"` |
| `dark_bg` | `False` | Dark background (great for glowing orbital renders) |
| `show_labels` | `True` | Show atom labels |

Returns a `plotly.graph_objects.Figure`. Call `.show()` to render in Jupyter or `.write_html("out.html")` to save.

---

## Examples

See the [`examples/`](examples/) directory for Jupyter notebooks:

- [`ethylene_from_pybest.ipynb`](examples/ethylene_from_pybest.ipynb) — ethylene MOs from PyBEST output
- [`cyclobutadiene_from_molden.ipynb`](examples/cyclobutadiene_from_molden.ipynb) — cyclobutadiene MOs from a Molden file

---

## Dependencies

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Plotly](https://plotly.com/python/)

---

## License

MIT
