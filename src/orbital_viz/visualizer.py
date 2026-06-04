from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from orbital_viz.gto import BasisGTO
from orbital_viz.utils import _COLOR, _SYMBOL, _VDW_R, cylinder_mesh, find_bonds, get_sphere_coords

# ── Camera angle helpers ───────────────────────────────────────────────────────

_DEFAULT_EYE = dict(x=1.6, y=1.6, z=1.2)


def _angles_to_eye(azimuth_deg: float, elevation_deg: float, r: float = 2.5) -> dict:
    """Convert spherical (azimuth, elevation) angles in degrees to a Plotly camera eye dict."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return dict(
        x=float(r * np.cos(el) * np.cos(az)),
        y=float(r * np.cos(el) * np.sin(az)),
        z=float(r * np.sin(el)),
    )


def _eye_to_angles(eye: dict) -> tuple[float, float]:
    """Return (azimuth_deg, elevation_deg) from a Plotly camera eye dict."""
    x, y, z = eye["x"], eye["y"], eye["z"]
    r = np.sqrt(x**2 + y**2 + z**2)
    elevation = float(np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(y, x)))
    return azimuth, elevation


# ── HTML export with live angle overlay ───────────────────────────────────────

_ANGLE_POST_SCRIPT = """
(function() {
    var gd = document.querySelector('.js-plotly-plot');
    if (!gd) return;

    var overlay = document.createElement('div');
    overlay.id = 'orbital-angle-overlay';
    overlay.style.cssText = (
        'position:absolute;bottom:8px;right:8px;padding:4px 10px;'
        'font:13px monospace;border-radius:4px;pointer-events:none;z-index:9999;'
        + '{OVERLAY_STYLE}'
    );
    overlay.textContent = 'Az: {AZ:.1f}°  El: {EL:.1f}°';
    gd.style.position = 'relative';
    gd.appendChild(overlay);

    gd.on('plotly_relayout', function(ev) {
        var cam = (ev['scene.camera'] || {}).eye
               || ((gd._fullLayout || {}).scene || {})._camera && gd._fullLayout.scene._camera.eye;
        if (!cam) {
            var sceneCamera = (gd.layout && gd.layout.scene && gd.layout.scene.camera);
            cam = sceneCamera && sceneCamera.eye;
        }
        if (!cam) return;
        var r = Math.sqrt(cam.x*cam.x + cam.y*cam.y + cam.z*cam.z);
        if (r < 1e-9) return;
        var el = Math.asin(Math.max(-1, Math.min(1, cam.z / r))) * 180 / Math.PI;
        var az = Math.atan2(cam.y, cam.x) * 180 / Math.PI;
        overlay.textContent = 'Az: ' + az.toFixed(1) + '°  El: ' + el.toFixed(1) + '°';
    });
})();
"""


def figure_to_html(fig: go.Figure, *, show_live_angles: bool = True, **to_html_kwargs) -> str:
    """
    Serialise *fig* to a self-contained HTML string.

    Parameters
    ----------
    fig : go.Figure
        Figure produced by :func:`plot_molecular_orbital`.
    show_live_angles : bool
        When ``True`` (default) a floating overlay in the bottom-right corner
        displays the current azimuth and elevation angles and updates in real
        time as the user rotates the molecule.
    **to_html_kwargs
        Forwarded verbatim to :meth:`plotly.graph_objects.Figure.to_html`.

    Returns
    -------
    str
        Complete HTML document ready to save as ``.html`` or embed in a page.
    """
    if not show_live_angles:
        return fig.to_html(**to_html_kwargs)

    # Detect background colour from figure so the overlay blends in
    bg = (fig.layout.paper_bgcolor or "#ffffff").lower()
    dark = bg not in ("#ffffff", "white", "rgb(255,255,255)", "")
    if dark:
        overlay_style = "background:rgba(0,0,0,0.55);color:#ffffff;border:1px solid #555;"
    else:
        overlay_style = "background:rgba(255,255,255,0.75);color:#111111;border:1px solid #bbb;"

    # Read the initial angles from the figure's camera
    try:
        eye = fig.layout.scene.camera.eye.to_plotly_json()
    except (AttributeError, TypeError):
        eye = _DEFAULT_EYE
    az0, el0 = _eye_to_angles(eye)

    post_script = (
        _ANGLE_POST_SCRIPT
        .replace("{OVERLAY_STYLE}", overlay_style)
        .replace("{AZ:.1f}", f"{az0:.1f}")
        .replace("{EL:.1f}", f"{el0:.1f}")
    )

    to_html_kwargs.setdefault("include_plotlyjs", "cdn")
    to_html_kwargs.setdefault("full_html", True)
    return fig.to_html(post_script=post_script, **to_html_kwargs)


# ── Main plotting function ─────────────────────────────────────────────────────

def plot_molecular_orbital(
    Basis: BasisGTO,
    mo_coeffs: np.ndarray,
    n: int,
    *,
    grid_points: int = 50,
    isovalue: float = 0.05,
    padding: float = 2.0,
    opacity: float = 0.45,
    atom_scale: float = 0.25,
    bond_radius: float = 0.08,
    title: Optional[str] = None,
    dark_bg: bool = False,
    show_labels: bool = True,
    azimuth: Optional[float] = None,
    elevation: Optional[float] = None,
    show_rotation_angles: bool = False,
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
    azimuth : float | None
        Initial camera azimuth angle in **degrees** (rotation in the x-y plane,
        measured from the +x axis counter-clockwise).  When supplied together
        with *elevation*, the molecule is shown from that viewpoint.
        If only one of the two is given the other retains its default value
        (≈ 45° azimuth, ≈ 28° elevation).
    elevation : float | None
        Initial camera elevation angle in **degrees** above the x-y plane.
        Positive values look down on the molecule; negative values look up.
    show_rotation_angles : bool
        When ``True``, a static annotation in the figure corner shows the
        current azimuth and elevation of the camera.  Automatically set to
        ``True`` whenever *azimuth* or *elevation* is specified.

    Returns
    -------
    plotly.graph_objects.Figure
        Call ``.show()`` in a Jupyter notebook cell to render it, or pass
        the figure to :func:`figure_to_html` to obtain a self-contained HTML
        file with a live angle overlay.
    """

    # ── 0. Resolve camera angles ───────────────────────────────────────────────
    _rotation_given = azimuth is not None or elevation is not None
    _default_az, _default_el = _eye_to_angles(_DEFAULT_EYE)
    _az = azimuth if azimuth is not None else _default_az
    _el = elevation if elevation is not None else _default_el
    camera_eye = _angles_to_eye(_az, _el) if _rotation_given else _DEFAULT_EYE
    _show_angles = show_rotation_angles or _rotation_given

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
    for i, j in find_bonds(coords, atoms):
        xc, yc, zc, ti, tj, tk = cylinder_mesh(
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

        # Actual physical radius in data units (Angstroms)
        rvdw = _VDW_R.get(z_int, 1.0) * atom_scale 
        symbol = _SYMBOL.get(z_int, str(z_int))

        # 1. Create the 3D Ball (Surface)
        sph_x, sph_y, sph_z = get_sphere_coords(xyz, rvdw)

        traces.append(
            go.Surface(
                x=sph_x, y=sph_y, z=sph_z,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=1.0,  # Set to 1.0 for solid 3D balls
                name=f"{symbol}{idx + 1}",
                hoverinfo="name"
            )
        )

        # 2. Create the Label (Scatter3d)
        if show_labels:
            traces.append(
                go.Scatter3d(
                    x=[xyz[0]],
                    y=[xyz[1]],
                    z=[xyz[2]],
                    mode="text",
                    text=[symbol],
                    textfont=dict(
                        size=14,
                        color="white" if dark_bg else "#111111",
                        family="Arial Black" # Makes the symbol pop
                    ),
                    showlegend=False, # Hide the text trace from legend to avoid duplicates
                    hoverinfo="skip"
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
            camera=dict(eye=camera_eye),
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

    if _show_angles:
        fig.add_annotation(
            text=f"Az: {_az:.1f}°  El: {_el:.1f}°",
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.01,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12, color=tc, family="monospace"),
            bgcolor="rgba(0,0,0,0.45)" if dark_bg else "rgba(255,255,255,0.7)",
            bordercolor=axc,
            borderwidth=1,
        )

    return fig