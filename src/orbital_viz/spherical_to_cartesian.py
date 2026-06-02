import numpy as np
from scipy.linalg import block_diag

from orbital_viz.utils import get_l_transformation


def spherical_to_cartesian_matrix(C_sph: np.ndarray, shell_l_list: list) -> np.ndarray:
    """
    Transform MO coefficients from spherical back to Cartesian basis.

    Parameters
    ----------
    C_sph : np.ndarray, shape ``(N_sph, N_mo)``
    shell_l_list : list of int
        Angular momentum *l* for each shell.

    Returns
    -------
    np.ndarray, shape ``(N_cart, N_mo)``
    """
    T = block_diag(*[get_l_transformation(l) for l in shell_l_list])
    return T.T @ C_sph
