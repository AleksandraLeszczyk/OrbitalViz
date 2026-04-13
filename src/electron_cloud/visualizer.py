from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from electron_cloud.gto import BasisGTO
from electron_cloud.utils import _COLOR, _SYMBOL, _VDW_R, _cylinder_mesh, _find_bonds

def plot_molecular_orbital(
    Basis: BasisGTO,
    mo_coeffs: np.ndarray,
    n: int,
    *,
    grid_points: int = 50,
    isovalue: float = 0.05,
    padding: float = 2.0,
    opacity: float = 0.45,
    atom_scale: float = 0.5,
    bond_radius: float = 0.08,
    title: Optional[str] = None,
    dark_bg: bool = False,
) -> go.Figure:
    """
    Render the *n*-th molecular orbital as an interactive Plotly 3-D figure.

    Positive ψ lobes are drawn in **blue**, negative lobes in **gold**.
    Atoms are CPK spheres; bonds are grey cylinders inferred from distances.

    Parameters
    ----------
    Basis : BasisGTO
        Basis set / molecular geometry.
    mo_coeffs : np.ndarray, shape ``(n_basis, n_mo)``
        MO coefficient matrix  C  (column-major: column *n* = MO *n*).
    n : int
        Zero-based index of the MO to visualise.
    grid_points : int
        Number of grid points along each Cartesian axis.  Increase for
        smoother isosurfaces; decrease for interactive speed (default 50).
    isovalue : float
        |ψ| at which isosurfaces are drawn.  If you see no lobes, try a
        smaller value; if the surface is too diffuse, increase it.
    padding : float
        Extra space (Å) added around the molecule bounding box.
    opacity : float
        Isosurface opacity: 0 = fully transparent, 1 = fully opaque.
    atom_scale : float
        Multiplicative scale applied to VDW radii when sizing atom spheres.
    bond_radius : float
        Cylinder radius for bond sticks [Å].
    title : str | None
        Figure title.  Defaults to ``"MO #n"``.
    dark_bg : bool
        Dark background (great for glowing orbitals) or white.

    Returns
    -------
    plotly.graph_objects.Figure
        Call ``.show()`` in a Jupyter notebook cell to render it.
    """

    coords = Basis.coordinates  # (N_at, 3) Å
    atoms = Basis.atoms

    # ── 1. Rectilinear grid ────────────────────────────────────────────────
    lo = coords.min(axis=0) - padding
    hi = coords.max(axis=0) + padding
    axes = [np.linspace(lo[k], hi[k], grid_points) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")  # (G, G, G)

    # ── 2. Evaluate MO on grid ─────────────────────────────────────────────
    psi = Basis.evaluate_mo(mo_coeffs, n, X, Y, Z)
    pmax = float(np.abs(psi).max())
    if pmax < 1e-14:
        raise ValueError(
            "MO is essentially zero everywhere – please check the input "
            "(coordinates in Å, coefficients in AO basis, 0-based MO index)."
        )

    xf = X.ravel()
    yf = Y.ravel()
    zf = Z.ravel()
    vf = psi.ravel()

    # ── 3. Isosurfaces ─────────────────────────────────────────────────────
    def _iso_trace(
        v_lo: float, v_hi: float, hex_color: str, name: str
    ) -> go.Isosurface:
        return go.Isosurface(
            x=xf,
            y=yf,
            z=zf,
            value=vf,
            isomin=v_lo,
            isomax=v_hi,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=[[0.0, hex_color], [1.0, hex_color]],
            showscale=False,
            opacity=opacity,
            name=name,
            lighting=dict(
                ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.3, fresnel=0.4
            ),
            lightposition=dict(x=1000, y=1000, z=1000),
        )

    traces: list[go.BaseTraceType] = [
        _iso_trace(isovalue, pmax * 1.01, "#2166AC", f"ψ = +{isovalue:.3f}"),  # blue
        _iso_trace(-pmax * 1.01, -isovalue, "#FFD700", f"ψ = −{isovalue:.3f}"),  # gold
    ]

    # ── 4. Bond sticks (Mesh3d cylinders) ──────────────────────────────────
    for i, j in _find_bonds(coords, atoms):
        xc, yc, zc, ti, tj, tk = _cylinder_mesh(
            coords[i], coords[j], radius=bond_radius
        )
        traces.append(
            go.Mesh3d(
                x=xc,
                y=yc,
                z=zc,
                i=ti,
                j=tj,
                k=tk,
                color="#aaaaaa",
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
                showscale=False,
                showlegend=False,
                name="bond",
            )
        )

    # ── 5. Atom spheres (Scatter3d) ────────────────────────────────────────
    for idx, (z_at, xyz) in enumerate(zip(atoms, coords)):
        z_int = int(z_at)
        color = _COLOR.get(z_int, "#CCCCCC")
        rvdw = _VDW_R.get(0.3 * z_int, 0.77)
        symbol = _SYMBOL.get(z_int, str(z_int))
        traces.append(
            go.Scatter3d(
                x=[xyz[0]],
                y=[xyz[1]],
                z=[xyz[2]],
                mode="markers+text",
                marker=dict(
                    size=rvdw * atom_scale * 40,  # empirical px scaling
                    color=color,
                    symbol="circle",
                    line=dict(color="#333333", width=1),
                ),
                text=[symbol],
                textfont=dict(
                    size=13,
                    color="white" if dark_bg else "#111111",
                ),
                textposition="top center",
                name=f"{symbol}{idx + 1}",
                showlegend=True,
            )
        )
    # ── 6. Layout ──────────────────────────────────────────────────────────
    bg = "#0d0d1a" if dark_bg else "#ffffff"
    axc = "#bbbbbb" if dark_bg else "#DCD5D5"
    tc = "white" if dark_bg else "#111111"

    def _axis(label: str) -> dict:
        return dict(
            title=dict(text=label, font=dict(color=axc, size=11)),
            showbackground=False,
            showticklabels=False,
            gridcolor=axc,
            zeroline=False,
            showspikes=False,
        )
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=title or f"Molecular Orbital  #{n}",
            x=0.5,
            font=dict(size=18, color=tc, family="monospace"),
        ),
        scene=dict(
            bgcolor=bg,
            xaxis=_axis("x / Å"),
            yaxis=_axis("y / Å"),
            zaxis=_axis("z / Å"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        paper_bgcolor=bg,
        font=dict(color=tc),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0.45)" if dark_bg else "rgba(255,255,255,0.7)",
            bordercolor=axc,
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig