'''
Script for producing 1-dimensional tests of neutrino matter effects with the SME

Simon Hilding-Nørkjær
'''

import sys, os, collections, datetime
from astropy.time import Time
import time as time_module
import numpy as np
import itertools
from scipy.spatial.transform import Rotation

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import * #calc_path_length_from_coszen, get_coszen_from_path_length
from deimos.utils.coordinates import * #get_right_ascension_and_declination
from deimos.utils.constants import * 


def get_mixing_matrix(matrix, E_eV=1):
    """
    Calculates the mixing matrix and eigenvalues of a given Hamiltonian matrix.
    """
    # Rescale matrix to account for energy dependence
    matrix = matrix*2*E_eV

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Check if the matrix is diagonalizable
    if not np.all(np.iscomplex(eigenvalues) | np.isreal(eigenvalues)):
        raise ValueError("Matrix is not diagonalizable.")

    return sorted_eigenvectors, eigenvalues


def diagonalize_H2E(H2E, deg=True):
    """
    Returns all mixing angles that diagonalize the matrix H2E.
    """
    # Calculate the mixing matrix
    mixing_matrix, eigenvalues = get_mixing_matrix(H2E)

    # Initialize a list to store the diagonalizing angles
    diagonalizing_angles = []
    diagonalized_matrices = []

    # Try all mixing orders and signs
    n = mixing_matrix.shape[0]
    signs = [-1, 1]
    column_orders = list(itertools.permutations(range(n)))

    for order in column_orders:
        for sign_combination in itertools.product(signs, repeat=n):
            modified_matrix = mixing_matrix[:, order] * np.array(sign_combination)
            angles = pmns_angles_from_PMNS(matrix=modified_matrix, deg=False)

            if angles is not None:
                theta12, theta13, theta23 = angles

                if 0 <= theta12 <= np.pi/2 and 0 <= theta13 <= np.pi/2 and 0 <= theta23 <= np.pi/2:
                    result_array = np.dot(modified_matrix.conjugate().T, np.dot(H2E, modified_matrix))
                    # Set values close to zero to zero based on the tolerance
                    tolerance = 1e-9
                    result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

                    if np.allclose(result_array, np.diag(np.diagonal(result_array))):
                        # Assert that result_array is diagonal
                        diagonalizing_angles.append((theta12, theta13, theta23))
                        diagonalized_matrices.append(result_array)

    if deg:
        return [np.rad2deg(angles) for angles in diagonalizing_angles], diagonalized_matrices
    else:
        return diagonalizing_angles, diagonalized_matrices


def pmns_angles_from_PMNS(matrix, deg=True):
    """
    Returns the mixing angles.
    """
    #assert is_unitary(matrix)

    # Convert the matrix to a rotation object
    rotation = Rotation.from_matrix(matrix)
    
    # Try canonical rotation order
    order = 'zyx'
    angles = rotation.as_euler(order, degrees=True)

    # Correct for wrong sign of theta23 and theta12
    angles[0] = -angles[0]
    angles[2] = -angles[2]
        
    # Extract the angles of rotation around x, y, and z axes
    theta12, theta13, theta23 = angles 

    # Check whether angles lie in first quadrant
    #if np.all(angles >= 0) and np.all(angles <= 90):
    #    pass
    #else:
    #    raise Exception("Angles are not in the first quadrant. The PMNS matrix might have been calculated using a wrong order. The calculated angels are ", angles)

    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)
    

#
# Main
#

if __name__ == "__main__" :

    print("Running", __file__)
    
    #
    # Define basic system
    #
    
    detector = "arca"     # "arca" or "dune"

    initial_flavor = 1          # numu survival
    nubar = False             # neutrino or antineutrino

    E_array_type = True
    E_GeV = np.array([10000.,20000.])
    E_node = 0

    # cosz_deg = np.linspace(-1,0, num=149)
    # baseline = calc_path_length_from_coszen(cosz_deg)
    baseline = np.linspace(0,EARTH_DIAMETER_km, num=1000)

    directional = True
    atmospheric = False
   
    a_magnitude_eV = 4e-13 # Overall strength of a component
    c_magnitude = 0#2e-26 # Overall strength of c component

    flavor_structure =    np.array([0., 0., 1.])         # numu->nutau
    field_direction_structure = np.array([0., 1., 0.])        # Orientation of field

    neutrino_offset_from_field_direction_RA_deg = 180
    neutrino_offset_from_field_direction_DEC_deg = 0

    # Choose solver (nusquids or deimos)
    solver = "nusquids"
    sme_basis = "mass"

    earth_model = "prem" # "prem" or "simple"


    #
    # Matter models
    #


    # read EARTH_MODEL_PREM.dat file and extract layer endpoints, matter densities and electron fractions
    # scale layer endpoints to km and mirror the arrays to get the full earth model as a function of baseline
    layer_endpoint_array_as_fraction_of_earth_radius = np.loadtxt("EARTH_MODEL_PREM.dat", usecols=0)
    matter_density_g_per_cm3_array = np.loadtxt("EARTH_MODEL_PREM.dat", usecols=1)
    electron_fraction_array = np.loadtxt("EARTH_MODEL_PREM.dat", usecols=2)

    layer_endpoint_km_array = layer_endpoint_array_as_fraction_of_earth_radius*EARTH_RADIUS_km
    layer_endpoint_km_array = np.append(layer_endpoint_km_array, EARTH_RADIUS_km+layer_endpoint_km_array)

    matter_density_g_per_cm3_array = np.flip(matter_density_g_per_cm3_array)
    matter_density_g_per_cm3_array = np.append(matter_density_g_per_cm3_array, np.flip(matter_density_g_per_cm3_array))

    electron_fraction_array = np.flip(electron_fraction_array)
    electron_fraction_array = np.append(electron_fraction_array, np.flip(electron_fraction_array))

    # Definition of the simple earth model (3 layers)
    if earth_model == "simple":
            
            inner_core_thickness_km = 1221.5*2          #x2 because 1221 is the radius
            outer_core_thicknes_km = 3480.0-1221.5      #3480 is the outer core radius
            mantle_thickness_km = 5701.0-3480.0         #5701 is the mantle radius
            transition_and_crust_thickness_km = 6371.0-5701.0 #6371 is the earth radius

            #for simplicity is the transition and crust layer treated as part of the mantle (quite thin layers) #TODO maybe implement layers for both transition and crust, as densities vary a lot
            mantle_thickness_km = mantle_thickness_km + transition_and_crust_thickness_km

            # define the matter densities (g/cm3) and electron fractions for each of the earths layers

            mask_mantle = layer_endpoint_km_array < mantle_thickness_km
            mask_outer_core = (layer_endpoint_km_array > mantle_thickness_km) & (layer_endpoint_km_array < mantle_thickness_km+outer_core_thicknes_km)
            mask_inner_core = (layer_endpoint_km_array > mantle_thickness_km+outer_core_thicknes_km) & (layer_endpoint_km_array < mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km)
            
            matter_density_mantle = np.average(matter_density_g_per_cm3_array[mask_mantle])
            matter_density_outer_core = np.average(matter_density_g_per_cm3_array[mask_outer_core])
            matter_density_inner_core = np.average(matter_density_g_per_cm3_array[mask_inner_core])



            layer_endpoint_km_array = np.array([ mantle_thickness_km, 
                                                mantle_thickness_km+outer_core_thicknes_km, 
                                                mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km, 
                                                mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km+outer_core_thicknes_km, 
                                                mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km+outer_core_thicknes_km+mantle_thickness_km])
            matter_density_g_per_cm3_array = np.array([matter_density_mantle, matter_density_outer_core, matter_density_inner_core, matter_density_outer_core, matter_density_mantle])
            electron_fraction_array = np.array([0.5, 0.5, 0.5, 0.5, 0.5])


    layer_matter_model = "layers"
    layer_matter_kwargs = {
                        "layer_endpoint_km":layer_endpoint_km_array,
                        "matter_density_g_per_cm3":matter_density_g_per_cm3_array, 
                        "electron_fraction":electron_fraction_array
                        }


    # layer radii and thicknesses from the simple earth model: Originally from PREM paper.
    inner_core_radius = 1221.5
    outer_core_radius = 3480.0
    mantle_radius = 5701.0
    earth_radius_km = 6371.0

    inner_core_thickness_km = inner_core_radius*2                               #x2 because 1221 is the radius
    outer_core_thicknes_km = outer_core_radius - inner_core_radius              #3480 is the outer core radius
    mantle_thickness_km = mantle_radius - outer_core_radius                     #5701 is the mantle radius
    transition_and_crust_thickness_km = earth_radius_km - mantle_radius         #6371 is the earth radius

    # currently transition and crust layer is treated as part of the mantle
    mantle_thickness_km = mantle_thickness_km + transition_and_crust_thickness_km




    #
    # Create calculators
    # 
    
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"


    calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)


    if field_direction_structure[0]   != 0: field_direction_coords = (0,0)
    elif field_direction_structure[1] != 0: field_direction_coords = (90,0)
    elif field_direction_structure[2] != 0: field_direction_coords = (0,90)
    else: raise Exception("Direction structure must be a unit vector")
    direction_string = np.array(["x","y","z"])[field_direction_structure.astype(bool)]

    a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in field_direction_structure ])
    ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in field_direction_structure ])

    time = "July 16, 1999, 10:30"


    #
    # MAIN LOOP
    #

    
    # Neutrino direction
    ra_deg = field_direction_coords[0] + neutrino_offset_from_field_direction_RA_deg
    dec_deg = field_direction_coords[1] + neutrino_offset_from_field_direction_DEC_deg
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    neutrino_coords = (ra_deg, dec_deg)


    #
    # Calculate oscillation probabilities:
    #


    # Define args to osc prob calc
    calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV":E_GeV,
        #"ra_rad":ra_rad,
        #"dec_rad":dec_rad,
        # "time":time,
    }

    # Printt layer matter model
    print(layer_matter_model, layer_matter_kwargs)

    # Oscillation probabilities
    layer_osc_prob = np.zeros((3, len(baseline)))
    # Simple Earth case
    calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    calculator.set_matter(layer_matter_model, **layer_matter_kwargs)
    osc_prob_loop = calculator.calc_osc_prob(distance_km = baseline, **calc_kw)
    layer_osc_prob[:,:] = osc_prob_loop[E_node].T       # Transpose and select energy node


    # Check that probabilities sum to 1 for each data point
    assert np.isclose( np.sum(layer_osc_prob[:,:]), len(layer_osc_prob[0,:]), atol=1e-10)


    # Mixing angles 
    angles_over_baseline = np.zeros((3, len(baseline)))
    counter = 0

    # Set SME again to get the Hamiltonian
    calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    calculator.set_matter(layer_matter_model, **layer_matter_kwargs)
    

    # Get Hamiltonian vs time (e.g. distance)
    for x in baseline:
        H = calculator.nusquids.GetHamiltonianAtTime(x, 0, 0, True) # args: [x, E node, rho (nu vs nubar), flavor (True) or mass (False) basis]
        H = np.array(H[:,:,0] + H[:,:,1]*1j, dtype=np.complex128) # This properly builds the real and imaginary components
        # print("\n x = %0.2f km \n%s" % (x, H*2*E_GeV[0]*1e9))

        # Calculate all mixing angles
        all_angles, mixing_matrices = diagonalize_H2E(H*2*E_GeV[0]*1e9, deg=True)

        angles_over_baseline[0,counter] = all_angles[0][0]
        angles_over_baseline[1,counter] = all_angles[0][1]
        angles_over_baseline[2,counter] = all_angles[0][2]

        counter += 1

    #
    # Plot oscillation probabilities
    #


    labes = [r"$\nu_\mu \rightarrow \nu_e$", r"$\nu_\mu \rightarrow \nu_\mu$", r"$\nu_\mu \rightarrow \nu_\tau$"]

    fig, axs = plt.subplots(2, 1, figsize=(7, 10), sharex=True, gridspec_kw={'height_ratios': [8, 2]})

    # Oscillation probabilities subplot
    ax = axs[0]
    ax.axvspan(0, EARTH_DIAMETER_km, color="yellow", alpha=0.5, label="Mantle")
    ax.axvspan(earth_radius_km - outer_core_radius, earth_radius_km + outer_core_radius, color="orange", alpha=0.5, label="Outer core")
    ax.axvspan(earth_radius_km - inner_core_radius, earth_radius_km + inner_core_radius, color="red", alpha=0.3, label="Inner core")
    ax.axvspan(0, EARTH_DIAMETER_km, color="white", alpha=0.5)
    ax.plot(baseline, layer_osc_prob[0, :], c='k', lw=3.5, label=f"P(" + labes[0] + ")", zorder=10)
    ax.plot(baseline, layer_osc_prob[1, :], c='darkorchid', ls="--", lw=2, alpha=1, label=f"P(" + labes[1] + ")")
    ax.plot(baseline, layer_osc_prob[2, :], c='dodgerblue', ls="-.", lw=2, alpha=1, label=f"P(" + labes[2] + ")")
    ax.set(xlim=(baseline[0], baseline[-1]), ylim=(-0.03, 1.03))
    ax.set_ylabel("Oscillation Probability", fontsize=20)
    #ax.set_xlabel("Baseline [km]", fontsize=20)
    if earth_model == "prem":
        ax.set_title("200 layer PREM Earth Model", fontsize=20)
    elif earth_model == "simple":
        ax.set_title("3 layer Simple Earth Model", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.axvline(x=earth_radius_km, color="black", alpha=0.2)  # label="Center")
    ax.legend(fontsize=13, ncol=2, loc="upper right")

    print(angles_over_baseline)
    # Cosine of mixing angles subplot
    ax = axs[1] 
    ax.axvspan(0, EARTH_DIAMETER_km, color="yellow", alpha=0.5)
    ax.axvspan(earth_radius_km - outer_core_radius, earth_radius_km + outer_core_radius, color="orange", alpha=0.5)
    ax.axvspan(earth_radius_km - inner_core_radius, earth_radius_km + inner_core_radius, color="red", alpha=0.3)
    ax.axvspan(0, EARTH_DIAMETER_km, color="white", alpha=0.5)
    ax.plot(baseline, np.cos(np.deg2rad(angles_over_baseline[0])), c='green', lw=2, label=r"$\cos(\theta_{12})$")
    ax.plot(baseline, np.cos(np.deg2rad(angles_over_baseline[1])), c='blue', lw=2, label=r"$\cos(\theta_{13})$")
    ax.plot(baseline, np.cos(np.deg2rad(angles_over_baseline[2])), c='red', lw=2, label=r"$\cos(\theta_{23})$")
    ax.set(xlim=(baseline[0], baseline[-1]), ylim=(-1.03, 1.03))
    ax.set_ylabel(r"$\cos(\theta_{ij})$", fontsize=20)
    ax.set_xlabel("Baseline [km]", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.axvline(x=earth_radius_km, color="black", alpha=0.2)  # label="Center")
    ax.legend(fontsize=13, loc="upper right")

    fig.suptitle("SME: a = {} eV, c = {} // LIV-offset:{}deg // E={}GeV".format(a_magnitude_eV, c_magnitude, neutrino_offset_from_field_direction_RA_deg, E_GeV[E_node]))

    # fig , axs = plt.subplots(3, 1, figsize=(12,12))

    # for i , ax in enumerate(axs):
    #     ax.axvspan(0,EARTH_DIAMETER_km, color="yellow", alpha=0.2, label="Mantle")
    #     ax.axvspan(earth_radius_km-outer_core_radius,earth_radius_km+outer_core_radius, color="orange", alpha=0.2, label="Outer core")
    #     ax.axvspan(earth_radius_km-inner_core_radius,earth_radius_km+inner_core_radius, color="red", alpha=0.2, label="Inner core")
    #     ax.plot(baseline, layer_osc_prob[i,:],c='k', lw=2, label=f"P("+labes[i]+")")
    #     ax.set(xlim=(baseline[0],baseline[-1]), ylim=(-0.03,1.03), ylabel="Oscillation Probability", xlabel="Baseline [km]")
    #     ax.set_title("Simple Earth")
    #     ax.axvline(x=earth_radius_km, color="black", alpha=0.4, label="Center")
    #     ax.legend(fontsize=8)

    fig.suptitle("SME: a = {} eV, c = {} // LIV-offset:{}deg // E={}GeV".format(a_magnitude_eV, c_magnitude, neutrino_offset_from_field_direction_RA_deg, E_GeV[E_node] ))
    fig.tight_layout()      


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
