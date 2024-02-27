'''
Script for producing 1-dimensional tests of neutrino matter effects with the SME

Simon Hilding-Nørkjær
'''



import sys, os, collections, datetime
from astropy.time import Time
import time as time_module
import numpy as np
import matplotlib as mpl

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import * #calc_path_length_from_coszen, get_coszen_from_path_length
from deimos.utils.coordinates import * #get_right_ascension_and_declination
from deimos.utils.constants import * 


#
# Main
#

if __name__ == "__main__" :


    #
    # Define basic system
    #
    detector = "arca"     # "arca" or "dune"

    initial_flavor = 1        # numu survival
    nubar = False             # neutrino or antineutrino

    E_array_type = True
    E_GeV = np.array([10000.,20000.])
    E_node = 0


    baseline = np.linspace(0,EARTH_DIAMETER_km, num=200)

    directional = True
    atmospheric = False
   
  
    a_magnitude_eV_1 = 2e-13 # Overall strength of a component
    a_magnitude_eV_2 = 4e-13 # Overall strength of a component
    a_magnitude_eV_3 = 6e-13 # Overall strength of a component
    c_magnitude = 0 #2e-26 # Overall strength of c component

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





    # Create calculators
    # Create calculators
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"


    calculator1 = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    calculator2 = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    calculator3 = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)


    


    if field_direction_structure[0]   != 0: field_direction_coords = (0,0)
    elif field_direction_structure[1] != 0: field_direction_coords = (90,0)
    elif field_direction_structure[2] != 0: field_direction_coords = (0,90)
    else: raise Exception("Direction structure must be a unit vector")
    direction_string = np.array(["x","y","z"])[field_direction_structure.astype(bool)]

    a_eV_1 = np.array([ a_magnitude_eV_1*n*np.diag(flavor_structure) for n in field_direction_structure ])
    a_eV_2 = np.array([ a_magnitude_eV_2*n*np.diag(flavor_structure) for n in field_direction_structure ])
    a_eV_3 = np.array([ a_magnitude_eV_3*n*np.diag(flavor_structure) for n in field_direction_structure ])
    ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in field_direction_structure ])

    time = "July 16, 1999, 10:30"




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
        "ra_rad":ra_rad,
        "dec_rad":dec_rad,
    }



    layer_osc_prob1 = np.zeros((3, len(baseline)))
    layer_osc_prob2 = np.zeros((3, len(baseline)))
    layer_osc_prob3 = np.zeros((3, len(baseline)))

    calculator1.set_matter(layer_matter_model, **layer_matter_kwargs)
    calculator2.set_matter(layer_matter_model, **layer_matter_kwargs)
    calculator3.set_matter(layer_matter_model, **layer_matter_kwargs)


    calculator1.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV_1, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    osc_prob_loop1 = calculator1.calc_osc_prob(distance_km = baseline, **calc_kw)
    layer_osc_prob1[:,:] = osc_prob_loop1[E_node].T       # Transpose and select energy node

    calculator2.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV_2, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    osc_prob_loop2 = calculator2.calc_osc_prob(distance_km = baseline, **calc_kw)
    layer_osc_prob2[:,:] = osc_prob_loop2[E_node].T       # Transpose and select energy node

    calculator3.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV_3, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    osc_prob_loop3 = calculator3.calc_osc_prob(distance_km = baseline, **calc_kw)
    layer_osc_prob3[:,:] = osc_prob_loop3[E_node].T       # Transpose and select energy node



    # plot oscillation probabilities
    #

    fig, ax1= plt.subplots(figsize=(7,6))
    ax1.imshow(np.array([matter_density_g_per_cm3_array, matter_density_g_per_cm3_array]), cmap="inferno", extent=[0, EARTH_DIAMETER_km, -0.03,1.03], aspect="auto", alpha=0.8, zorder=0, vmin=0, vmax=15)


    fig, ax= plt.subplots(figsize=(9,7))

    #plot earth layers from 200 layer PREM as imshow background
    ax.imshow(np.array([matter_density_g_per_cm3_array, matter_density_g_per_cm3_array]), cmap="inferno", extent=[0, EARTH_DIAMETER_km, -0.03,1.03], aspect="auto", alpha=0.8, zorder=0, vmin=0, vmax=15)

    ax.plot(baseline, layer_osc_prob1[0,:], c='r',lw=3.5, label=rf"$a_L=${a_magnitude_eV_1}", zorder=10)
    ax.plot(baseline, layer_osc_prob2[0,:], c='b',lw=3.5, label=rf"$a_L=${a_magnitude_eV_2}", zorder=10)
    ax.plot(baseline, layer_osc_prob3[0,:], c='g',lw=3.5, label=rf"$a_L=${a_magnitude_eV_3}", zorder=10)

    ax.set(xlim=(baseline[0],baseline[-1]),ylim=(-0.03,1.03))
    ax.set_ylabel(r"$P(\nu_{\mu}\to\nu_{e})$", fontsize=20)
    ax.set_xlabel("Baseline [km]", fontsize=20)
    if earth_model == "prem":
        ax.set_title("200 layer PREM Earth Model", fontsize=20)
    elif earth_model == "simple":
        ax.set_title("3 layer Simple Earth Model", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=15, ncol=1, loc="upper right")

    fig.suptitle("SME: a = {} eV, c = {} // LIV-offset:{}deg // E={}GeV".format("varying", c_magnitude, neutrino_offset_from_field_direction_RA_deg, E_GeV[E_node] ))


    cbar_ax = fig.add_axes([0.912, 0.107, 0.025, 0.775])
    cbar = fig.colorbar(ax1.images[0], cax=cbar_ax, orientation="vertical", fraction=0.05, pad=0.05,norm=mpl.colors.Normalize(vmin=0, vmax=15))
    cbar.set_label(r"Density $\;[g/cm^3]$", fontsize=18)

    
    cbar.ax.tick_params(labelsize=16)



  

    plt.savefig(__file__.replace(".py",".pdf"), bbox_inches='tight')

    #
    # Done
    #

    # print("")
    # dump_figures_to_pdf( __file__.replace(".py",".pdf") )
