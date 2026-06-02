import numpy as np
from scipy.linalg import block_diag

from orbital_viz.utils import get_l_transformation


def cartesian_to_spherical_matrix(C_cart: np.ndarray, shell_l_list: list) -> np.ndarray:
    """
    Transform MO coefficients from Cartesian to spherical basis.

    Parameters
    ----------
    C_cart : np.ndarray, shape ``(N_cart, N_mo)``
    shell_l_list : list of int
        Angular momentum *l* for each shell.

    Returns
    -------
    np.ndarray, shape ``(N_sph, N_mo)``
    """
    T = block_diag(*[get_l_transformation(l) for l in shell_l_list])
    return T @ C_cart
