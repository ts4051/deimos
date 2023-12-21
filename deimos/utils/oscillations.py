'''
Some generally useful things for neutrino oscillations

Tom Stuttard
'''

import numpy as np

from deimos.utils.constants import EARTH_RADIUS_km, DEFAULT_ATMO_PROD_HEIGHT_km, DEFAULT_ATMO_DETECTOR_DEPTH_km, FERMI_CONSTANT

from deimos.utils.matrix_algebra import *


#
# Globals
#

# A canonical unit conversion for the oscillation frequency term
# Use this to convert from GeV and km to natural units
OSC_FREQUENCY_UNIT_CONVERSION = 1.267 #TODO More precise


#
# Functions
#

def calc_path_length_from_coszen(cz, r=EARTH_RADIUS_km, h=DEFAULT_ATMO_PROD_HEIGHT_km, d=DEFAULT_ATMO_DETECTOR_DEPTH_km) :
    '''
    Get the path length (baseline) for an atmospheric neutrino,
    given some cos(zenith angle).

    cz = cos(zenith) in radians, to be converted to path length in km
    r = Radius of Earth, in km
    h = Production height in atmosphere, in km
    d = Depth of detector (w.r.t. Earth's radius), in km
    '''
    return -r*cz +  np.sqrt( (r*cz)**2 - r**2 + (r+h+d)**2 )


def get_coszen_from_path_length(L,r=EARTH_RADIUS_km, h=DEFAULT_ATMO_PROD_HEIGHT_km, d=DEFAULT_ATMO_DETECTOR_DEPTH_km) :
    return (d**2+2*d*(h+r)+h**2+2*h*r-L**2)/(2*L*r)


def calc_disappearance_prob_2flav_vacuum(E_GeV, L_km, mass_splitting_eV2, theta_rad) :
    '''
    Simple analytic 2-flavor oscillation disappearance probability calculation
    '''
    return np.sin(2.*theta_rad)**2 * np.square(np.sin(1.27*mass_splitting_eV2*L_km/E_GeV))


def oscillation_averaged_transition_probability( pmns, initial_flavor, final_flavor ) :
    '''
    Calculate the oscillation-averaged transition probability
    e.g. diffuse astrophysical neutrinos

    Ref: https://arxiv.org/pdf/1810.00893.pdf equation 3
    '''
    prob = 0.
    for i in range(pmns.shape[0]) :
        prob += ( pmns[initial_flavor,i] * np.conj(pmns[initial_flavor,i]) ) * ( pmns[final_flavor,i] * np.conj(pmns[final_flavor,i]) )
    return prob.real


def calc_osc_wavelength_km_from_mass_splitting(E_GeV, mass_splitting_eV2) :
    '''
    Calculate the oscillation wavelength(s) from the neutrino energy and mass splittings

    WARNING: This is only valid for vacuum oscillations, use `calc_osc_wavelength_from_hamiltonian` to include matter effects

    Args:
      mass_splitting_eV2 : 
        2 flavor : [dm^2_21]
        3 flavor : [dm^2_21, dm^2_31, dm^2_32]
    '''
    return np.pi / ( OSC_FREQUENCY_UNIT_CONVERSION * mass_splitting_eV2 / E_GeV ) 


def diagonalize_hamiltonian(H): #TODO move to a file with the other matrix algebra functions

    E_ij, v_ij = np.linalg.eigh(H)

    return np.real( np.diag([ E_ij[0], E_ij[1], E_ij[2] ]) )


def calc_osc_wavelength_from_hamiltonian(H) :
    '''
    Calculate the oscillation wavelength(s) from a Hamiltonian

    The result will be in units of 1/[H], e.g, the inverse of whatever 
    units the Hamiltonian is provided in (normally [eV])

    This can include matter effects
    '''

    # Check if hamiltonian is diagonal and diagonalize if not:
    if 1:
        H = diagonalize_hamiltonian(H)

    osc_wavelengths = []
    
    assert isinstance(H, np.ndarray)
    assert is_square(H)
    num_states = H.shape[0]

    # 21 mass splitting
    osc_wavelength_21 = np.abs( 2. * np.pi / (H[1,1] - H[0,0]) ) # Added - H[0,0]
    osc_wavelengths.append(osc_wavelength_21)

    if num_states == 3 :

        # 31 mass splitting
        osc_wavelength_31 = np.abs( 2. * np.pi / (H[2,2] - H[0,0]) ) # Added - H[0,0] and np.abs()

        # 32 mass splitting (mass ordering dependent)
        if H[2,2] > 0. : #TODO What about matter potential screwing this check up?
            osc_wavelength_32 = np.abs( 2. * np.pi / (H[2,2] - H[1,1]) ) # Added np.abs()
        else :
            osc_wavelength_32 = np.abs( 2. * np.pi / (H[1,1] - H[2,2]) ) # Added np.abs()

        osc_wavelengths.extend([osc_wavelength_31, osc_wavelength_32])

    return np.array(osc_wavelengths)



def calc_effective_osc_params_in_matter_2flav(E_eV, mixing_angle_rad, mass_splitting_eV2, matter_density_g_per_cm3, electron_fraction) :
    '''
    Convert Vacuum mixing angle and mass splitting into the effective values in constant density matter

    This is for a 2-flavor system, where e is the 0th flavor

    Following https://cds.cern.ch/record/1114392/files/p159.pdf eqns (35-37)
    '''

    from deimos.density_matrix_osc_solver.density_matrix_osc_solver import get_electron_density_per_m3

    Ne = get_electron_density_per_m3(matter_density_g_per_cm3, electron_fraction)

    A = 2. * np.sqrt(2.) * FERMI_CONSTANT * Ne * E_eV / mass_splitting_eV2

    sin2_2theta = np.square( np.sin( 2. * mixing_angle_rad ) )
    cos_2theta = np.cos( 2. * mixing_angle_rad )

    matter_mass_splitting_eV2 = mass_splitting_eV2 * np.sqrt( sin2_2theta + np.square(cos_2theta - A) )

    sin2_2theta_m = sin2_2theta / ( sin2_2theta + np.square( cos_2theta - A ) )

    matter_mixing_angle_rad = np.arcsin( np.sqrt(sin2_2theta_m) ) / 2.

    return matter_mixing_angle_rad, matter_mass_splitting_eV2



#
# Test
#

def test() :

    #
    # Test coszen -> L
    #

    fig = Figure( ny=2, figsize=(6, 6) )
    
    prod_height = 15.
    detector_depth = 1.

    # Full range
    coszen = np.linspace(-1., 1., num=100)
    L = calc_path_length_from_coszen(cz=coszen, d=prod_height, h=prod_height)
    fig.get_ax(y=0).plot(L, coszen, color="orange")
    format_ax( ax=fig.get_ax(y=0), xlim=(0., np.max(L)), ylim=(coszen[0], coszen[-1]) )

    # Zoom on down-going region (see effects of prod height and detectr depth)
    coszen = np.linspace(0., 1., num=100)
    L = calc_path_length_from_coszen(cz=coszen, d=prod_height, h=prod_height)
    fig.get_ax(y=1).plot(L, coszen, color="orange")
    fig.get_ax(y=1).axvline(prod_height+prod_height, color="purple", linestyle="--", label="h+d", lw=2)
    format_ax( ax=fig.get_ax(y=1), xlim=(0., np.max(L)), ylim=(coszen[0], coszen[-1]) )

    fig.quick_format(xlabel=r"$L$ [km]", ylabel=r"$\cos(\theta_{\rm{zenith}})$")


    #TODO test `h` and `d`


    #
    # Done
    #

    # Dump figures to PDF
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )


if __name__ == "__main__" :

    test()