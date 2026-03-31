# ElectronCloud

Visualise Gaussian-Type Orbital wave-functions interactively with Plotly.

Quick-start in a Jupyter notebook
----------------------------------
    from mo_viewer import BasisGTO, plot_molecular_orbital, demo_h2, demo_water

    fig = demo_h2(mo_index=0)   # 1σg bonding orbital
    fig.show()

    fig = demo_water(mo_index=4)  # 1b1 lone-pair HOMO
    fig.show()

Architecture
------------
BasisGTO
    Parses a shell-based GTO basis into a flat list of Cartesian AO
    functions and pre-computes (norm × contraction) coefficients so that
    evaluation on a grid is a single vectorised loop.

plot_molecular_orbital(Basis, mo_coeffs, n, ...)
    1. Builds a regular 3-D grid around the molecule.
    2. Evaluates every AO on the grid, then contracts them with the MO
       coefficients to obtain ψ(r).
    3. Renders:
         • Blue  isosurface  at  ψ = +isovalue
         • Gold  isosurface  at  ψ = −isovalue
         • Grey  cylinder    for every inferred bond
         • CPK sphere        for every atom (Scatter3d)