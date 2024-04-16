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

    initial_flavor = 1          # numu survival
    nubar = False             # neutrino or antineutrino

    E_array_type = True
    E_GeV = np.array([1000.,20000.])
    E_node = 0

    directional = True
    atmospheric = True
   
    a_magnitude_eV = 5e-14 # Overall strength of a component
    c_magnitude = 0 #2e-26 # Overall strength of c component

    flavor_structure =    np.array([0., 0., 1.])         # numu->nutau
    field_direction_structure = np.array([0., 1., 0.])        # Orientation of field


    # Choose solver (nusquids or deimos)
    solver = "nusquids"
    sme_basis = "mass"
    

    # Create calculators
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in field_direction_structure ])
    ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in field_direction_structure ])


    calculatorIC = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    calculatorARCA = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)

  
    calculatorIC.set_matter("vacuum")
    calculatorIC.set_detector("icecube")

    calculatorARCA.set_matter("vacuum")
    calculatorARCA.set_detector("arca")





    # Define time array
    hour_array = np.arange(1,25,2)
    time_array = []
    for time in hour_array: time_array.append(f"July 16, 1999, {time}:00")
    time_array = np.array(time_array)
    print(time_array)






    # Define coordinates
    ra_deg = 90
    dec_deg = np.linspace(-90,90, num=100)
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)


    cosz_IC = np.zeros(len(dec_deg))
    P_IC = np.zeros((3,len(dec_deg)))   
    cosz_ARCA = np.zeros((len(dec_deg),len(time_array)))
    P_ARCA = np.zeros((3,len(dec_deg),len(time_array)))

    #loop over dec and time
    for dec_index, dec in enumerate(dec_deg):

        # Define args to osc prob calc
        calc_kw = {
            "initial_flavor":initial_flavor,
            "nubar" : nubar,
            "energy_GeV":E_GeV,
            "ra_rad":ra_rad,
            "dec_rad":dec_rad[dec_index],
        }

        for time_index, time in enumerate(time_array):
            print(f"Progress: {dec_index}/{len(dec_deg)} {time_index}/{len(time_array)}",end="\r")

            P_ARCA_value, coszen_ARCA_value, azimuth_ARCA = calculatorARCA.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, time=time, **calc_kw)
            cosz_ARCA[dec_index,time_index] = coszen_ARCA_value
            P_ARCA[:,dec_index,time_index] = P_ARCA_value[E_node,:]

        P_IC_value, coszen_IC_value, azimuth_IC = calculatorIC.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, time=time_array[0], **calc_kw)
        cosz_IC[dec_index] = coszen_IC_value
        P_IC[:,dec_index] = P_IC_value[E_node,:]



    #convert cosz to baseline
    baseline_ARCA = calc_path_length_from_coszen(cosz_ARCA)
    baseline_IC = calc_path_length_from_coszen(cosz_IC)


    # Plot

    colors = plt.cm.jet(hour_array/24)
    plt.rcParams.update({'font.size': 14})


    fig,ax = plt.subplots(3,1,figsize=(6,14) )

    for time_index, time in enumerate(time_array):
        if time_index+1 == len(time_array): break

        ax[0].plot(dec_deg,cosz_ARCA[:,time_index+1], color=colors[time_index])
        ax[1].plot(dec_deg,baseline_ARCA[:,time_index+1],color=colors[time_index])
        ax[2].plot(dec_deg, P_ARCA[1,:,time_index+1],color=colors[time_index])
        
    

    ax[0].plot(dec_deg,cosz_ARCA[:,0], label=f"ARCA",color=colors[5])
    ax[1].plot(dec_deg,baseline_ARCA[:,0], label=f"ARCA",color=colors[5])
    ax[2].plot(dec_deg, P_ARCA[1,:,0], label=f"ARCA",color=colors[5])


    ax[0].plot(dec_deg, cosz_IC,"--k", lw=2,  label="IC")
    ax[1].plot(dec_deg, baseline_IC,"--k",lw=2, label="IC")
    ax[2].plot(dec_deg, P_IC[1,:],"--k",lw=2, label="IC")


    ax[2].set(ylim=(-0.05,1.05))
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()


    ax[0].set_ylabel(r"$\cos(\theta_{Zenith})$",fontsize=16)
    ax[1].set_ylabel("Baseline [km]",fontsize=16)
    ax[2].set_ylabel(r"$P(\nu_\mu \to \nu_\mu)$",fontsize=16)
    for i in range(3): 
        ax[i].set_xlabel("Declination [deg]",fontsize=16)
        ax[i].set_xticks([-90,-45,0,45,90])
        ax[i].grid(alpha=0.2)




    # add time colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=24)

    cax1 = fig.add_axes([0.92, 0.655, 0.02, 0.226])
    cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=plt.cm.jet, norm=norm, orientation='vertical')
    cb1.set_label('Time [h]')
    cb1.set_ticks(np.arange(0,25,4))
    cb1.set_ticklabels(np.arange(0,25,4))

    cax2 = fig.add_axes([0.92, 0.382, 0.02, 0.226])
    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=plt.cm.jet, norm=norm, orientation='vertical')
    cb2.set_label('Time [h]')
    cb2.set_ticks(np.arange(0,25,4))
    cb2.set_ticklabels(np.arange(0,25,4))

    cax3 = fig.add_axes([0.92, 0.11, 0.02, 0.226])
    cb3 = mpl.colorbar.ColorbarBase(cax3, cmap=plt.cm.jet, norm=norm, orientation='vertical')
    cb3.set_label('Time [h]')
    cb3.set_ticks(np.arange(0,25,4))
    cb3.set_ticklabels(np.arange(0,25,4))


    fig.suptitle(f"a = {a_magnitude_eV} eV, E = {E_GeV[E_node]} GeV, Vacuum", fontsize=16)
    





    plt.savefig(__file__.replace(".py",".pdf"), bbox_inches='tight')

    #
    # Done
    #

    # print("")
    # dump_figures_to_pdf( __file__.replace(".py",".pdf") )