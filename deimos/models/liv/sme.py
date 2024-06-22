'''
Implementation of the Standard Model Extension (SME)

Janni Nikolaides, Tom Stuttard
'''

import numpy as np
from deimos.utils.matrix_algebra import dagger


# def get_sme_hamiltonian_isotropic(
#     # Neutrino properties
#     E,
#     # LIV field properties
#     a_eV,
#     c,
# ) :
#     '''
#     Calculate the effective Hamiltonian for the isotropic model of the Standard Model Extension (SME).
#     '''

#     H_eff = a_eV - (E * c)   #TODO higher order

#     return H_eff

def get_sme_hamiltonian(
    # Neutrino properties
    ra,
    dec,
    E,
    # LIV field properties
    a_eV_t=None,
    a_eV_x=None,
    a_eV_y=None,
    a_eV_z=None,
    c_tt=None,
    c_tx=None,
    c_ty=None,
    c_tz=None,
    c_xx=None,
    c_xy=None,
    c_xz=None,
    c_yy=None,
    c_yz=None,
    c_zz=None,
    # System properties
    num_states=3, # 3-nu system by default
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

    # Check neutrino direction
    assert np.isscalar(ra)
    assert np.isscalar(dec)
    assert (ra >= 0) and (ra <= 2 * np.pi)
    assert (dec >= -np.pi / 2) and (dec <= np.pi / 2)

    # Check operators
    operator_shape = (num_states, num_states)
    if a_eV_t is None :
        a_eV_t = np.zeros(operator_shape)
    assert np.shape(a_eV_t) == operator_shape
    if a_eV_x is None :
        a_eV_x = np.zeros(operator_shape)
    assert np.shape(a_eV_x) == operator_shape
    if a_eV_y is None :
        a_eV_y = np.zeros(operator_shape)
    assert np.shape(a_eV_y) == operator_shape
    if a_eV_z is None :
        a_eV_z = np.zeros(operator_shape)
    assert np.shape(a_eV_z) == operator_shape
    if c_tt is None :
        c_tt = np.zeros(operator_shape)
    assert np.shape(c_tt) == operator_shape
    if c_tx is None :
        c_tx = np.zeros(operator_shape)
    assert np.shape(c_tx) == operator_shape
    if c_ty is None :
        c_ty = np.zeros(operator_shape)
    assert np.shape(c_ty) == operator_shape
    if c_tz is None :
        c_tz = np.zeros(operator_shape)
    assert np.shape(c_tz) == operator_shape
    if c_xx is None :
        c_xx = np.zeros(operator_shape)
    assert np.shape(c_xx) == operator_shape
    if c_xy is None :
        c_xy = np.zeros(operator_shape)
    assert np.shape(c_xy) == operator_shape
    if c_xz is None :
        c_xz = np.zeros(operator_shape)
    assert np.shape(c_xz) == operator_shape
    if c_yy is None :
        c_yy = np.zeros(operator_shape)
    assert np.shape(c_yy) == operator_shape
    if c_yz is None :
        c_yz = np.zeros(operator_shape)
    assert np.shape(c_yz) == operator_shape
    if c_zz is None :
        c_zz = np.zeros(operator_shape)
    assert np.shape(c_zz) == operator_shape

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



def get_sme_state_matrix(
    p11=0., # or e_e
    p12=0., # or e_mu
    p13=0., # or e_tau
    p22=0., # or mu_mu
    p23=0., # or mu_tau
    p33=0., # or tau_tau
) :
    '''
    Return the SME flavor/mass state matrix structure
    '''

    return np.array([
        [ p11,            p12,           p13 ], 
        [ np.conj(p12),   p22,           p23 ], 
        [ np.conj(p13),   np.conj(p23),  p33 ], 
    ])

