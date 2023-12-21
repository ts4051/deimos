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

    H_eff = a_eV + (E * c)   #TODO higher order

    return H_eff


def get_sme_hamiltonian_directional(
#amplitudes_effective_hamiltonian_sme(
    # Neutrino properties
    ra,
    dec,
    E,
    # M,
    # LIV field properties
    a_eV_x,
    a_eV_y,
    a_eV_z,
    c_tx,
    c_ty,
    c_tz,
    # e_m_x,
    # e_m_y,
    # e_m_z,
):
    """
    Calculate the effective Hamiltonian for the vector model of the Standard Model Extension (SME).

    The function calculates the effective Hamiltonian for the "vector model" of the SME. 
    It takes into account observer-dependent angles and amplitudes of the Lorentz-invariance violating physics
    specified by sme_a and sme_c. The result is the effective Hamiltonian at a given location for a neutrino source 
    located at (altitude, azimuth) on the sky and for a neutrino with energy E and propagating over a distance L.
    
    SME parameters are conventionally reported in the sun-centered celestial equatorial frame
    For simplicity, we approximate this frame to coincide with the earth-centered celestial equatorial frame
    Notes:
        Angles are to be provided in rad
    Returns: 
        3x3 matrix in flavor basis as ndarray
    References:
        - arXiv:hep-ph/0406255
        - arXiv:1010.4096 - note that this paper is now known to be wrong
    """

    #TODO Is this formalism dependent on the baseline being much smaller than the osc length (see https://arxiv.org/pdf/hep-ph/0406255.pdf page 2)? If so, this is a problem for atmo.....

    # print(a_eV_x, a_eV_y, a_eV_z, c_tx, c_ty, c_tz,'----')

    #
    # Coordinates
    #

    # celestial colatitude and longitude
    theta = np.pi/2 + dec 
    phi = ra 
    
    # spherical coordinates unit propagation vectors
    
    # r vector
    NX = np.sin(theta) * np.cos(phi)
    NY = np.sin(theta) * np.sin(phi)
    NZ = -np.cos(theta)
    
    # theta vector
    ThetaX = np.cos(theta)*np.cos(phi)
    ThetaY = np.cos(theta)*np.sin(phi)
    ThetaZ = -np.sin(theta)  
    
    # phi vector
    PhiX = -np.sin(phi)
    PhiY = np.cos(phi)
    PhiZ = 0
    
    # Polarization vector       #TODO doesn't seem to be used anywhere
    # PolX = 1/np.sqrt(2)*(ThetaX+1j*PhiX)
    # PolY = 1/np.sqrt(2)*(ThetaY+1j*PhiY)
    # PolZ = 1/np.sqrt(2)*(ThetaZ+1j*PhiZ)
    

    #
    # Amplitudes...
    #

    # the right ascension and declination of a neutrino are measured when detected. 
    # To account for the rotation of the earth during propagation we need to rotate
    # by R_z(-omega_sid*T_sid)
    # If ra and dec of source are known change sign of As. (As = -As)
    

    #
    # Mass independent operators
    #
    
    # Amplitude to be multiplied with sin(omega_sid L)
    As0 = -NY * a_eV_x + NX * a_eV_y
    As1 = + 2 * NY * c_tx - 2 * NX * c_ty
    As = As0 + E * As1

    # Amplitude to be multiplied with cos(omega_sid L)
    Ac0 = - NX * a_eV_x - NY * a_eV_y
    Ac1 = 2 * NX * c_tx + 2 * NY * c_ty
    Ac = Ac0 + E * Ac1

    #
    # Mass dependent operators
    #

    #TODO re-integrate this
    # Ac0_e_m = 1/2*(NX*e_m_x+NY*e_m_y)*dagger(M)+1/2*(M)*(NX*e_m_x+NY*e_m_y)
    # Ac += Ac0_e_m


    #
    # Put it all together
    #

    const = NZ * a_eV_z #TODO explain

    H_eff = Ac + const #TODO explain

    #TODO long baseline correction?

    return H_eff
