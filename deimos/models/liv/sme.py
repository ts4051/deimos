'''
Implementation of the Standard Model Extension (SME)

Janni Nikolaides, Tom Stuttard
'''

import numpy as np
from deimos.utils.matrix_algebra import dagger


def get_sme_hamiltonian_isotropic(
    # Neutrino properties
    E,
    # LIV field properties
    a_eV,
    c,
) :
    '''
    Calculate the effective Hamiltonian for the isotropic model of the Standard Model Extension (SME).
    '''

    H_eff = a_eV - (E * c)   #TODO higher order

    return H_eff

def get_sme_hamiltonian_directional(
    # Neutrino properties
    ra,
    dec,
    E,
    # LIV field properties
    a_eV_t=0,
    a_eV_x=0,
    a_eV_y=0,
    a_eV_z=0,
    c_tt=0,
    c_tx=0,
    c_ty=0,
    c_tz=0,
    c_xx=0,
    c_xy=0,
    c_xz=0,
    c_yy=0,
    c_yz=0,
    c_zz=0,
):
    """
    Calculate the effective Hamiltonian for the vector model of the Standard Model Extension (SME).

    The function calculates the effective Hamiltonian for the "vector model" of the SME. 
    It takes into account amplitudes of the Lorentz-invariance violating physics
    specified by sme_a and sme_c. The result is the effective Hamiltonian for a neutrino source 
    located at (ra, dec) on the sky and for a neutrino with energy E.

    Parameters:
        ra (float): Right ascension in radians.
        dec (float): Declination in radians.
        E (float): Neutrino energy in GeV.
        a_eV_t, a_eV_x, a_eV_y, a_eV_z (float): dimension-3, mass-independent, CPT-odd, SME coefficients a in eV.
        c_tt, c_tx, c_ty, c_tz, c_xx, c_xy, c_xz, c_yy, c_yz, c_zz (float): dimension-4, mass-independent, CPT-even, SME coefficients c.

    Returns:
        np.ndarray: 3x3 effective Hamiltonian matrix.

    References:
        - arXiv:hep-ph/0406255
    """

    # Convert celestial coordinates to spherical coordinates
    theta = np.pi / 2 - dec  # Colatitude
    phi = ra  # Longitude

    # Calculate the unit propagation vectors in spherical coordinates
    p_x = -np.sin(theta) * np.cos(phi)
    p_y = -np.sin(theta) * np.sin(phi)
    p_z = -np.cos(theta) 

    # Calculate the mass-independent operators
    H_eff = (
        a_eV_t - (a_eV_x * p_x + a_eV_y * p_y + a_eV_z * p_z)
        - E * (0.5 * c_tt * (3 - p_z * p_z) + 0.5 * c_zz * (-1 + 3 * p_z * p_z) - 2 * c_tz * p_z)
        + 2 * E * (c_tx * p_x + c_ty * p_y - c_xz * p_x * p_z - c_yz * p_y * p_z)
        - E * (c_xx * p_x * p_x + c_yy * p_y * p_y + 2 * c_xy * p_x * p_y)
    )

    return H_eff