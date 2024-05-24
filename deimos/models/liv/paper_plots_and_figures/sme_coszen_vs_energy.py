'''
SME neutrino oscillations plot of coszen vs energy

Simon Hilding-Nørkjær
'''

import sys, os, collections
import matplotlib.pyplot as plt
import numpy as np

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import calc_path_length_from_coszen, get_coszen_from_path_length


#
# Main
#

if __name__ == "__main__" :

    #
    # Create model
    #

    # Choose solver
    solver = "nusquids"
    # solver = "deimos"

    # Initial definitions
    energy_range_GeV = np.geomspace(1, 1e5, num=151)
    coszen_range = np.linspace(-1,1, num=149)
    initial_flavor = 1      #mu
    nubar = False

    sme_on = True
    directional=True     #TODO Reimplement directional=False
    atmospheric=False
    interactions=False
    matter_model="vacuum" # "earth", "constant", "vacuum"

    ra_rad = 90.
    dec_rad = 0.

    basis="mass"

    # Define LIV coefficients as matrices 

    a_magnitude_eV = 2e-13 # Overall strength of a component
    c_magnitude = 1e-26 # Overall strength of c component


    flavor_structure = np.array([0.0, 0.0,1.]) # not Equally shared between flavors (diagonal)

    if directional :
        direction_structure = np.array([0., 1., 0.]) # Orient field in x direction
        a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in direction_structure ])
        ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in direction_structure ])
        null_operator = np.zeros( (3, 3, 3) )

    else :
        a_eV = a_magnitude_eV * np.diag(flavor_structure)
        ct = c_magnitude * np.diag(flavor_structure)
        null_operator = np.zeros( (3,3) )



    #
    # Create calculators
    #


    kw =  {}
    if solver == "nusquids":
        kw = {"energy_nodes_GeV":energy_range_GeV,
              "interactions":interactions,}
        kw["nusquids_variant"] = "sme"


    calculator = OscCalculator(
        tool=solver,
        atmospheric=atmospheric,
        # num_neutrinos=3,
        **kw
        )
    
    calculator.set_matter(matter_model)

    # if sme_on:
    #     calculator.set_sme(directional=directional,basis=basis,a_eV=a_eV,c=ct,e=null_operator)
    # else:
    #     calculator.set_std()


    
    baseline = calc_path_length_from_coszen(coszen_range)



    #
    # Calculate oscillation probabilities
    #

    # calc_kw = {
    #             "initial_flavor":flavor_initial,
    #             "nubar" : False,
    #             "energy_GeV":energy_range_GeV,
    #             # "distance_km":baseline,
    #             "coszen":coszen_range,
    #             "ra_rad":ra_rad,
    #             "dec_rad":dec_rad,
    #         }



    # osc_probs = calculator.calc_osc_prob_sme(basis=basis,a_eV=a_eV,**calc_kw)
    # print(np.shape(osc_probs))
    # osc_probs = osc_probs.T
    # nue = osc_probs[0]
    # numu = osc_probs[1]
    # nutau = osc_probs[2]

    # print(np.shape(nue))

 
    # Define args to osc prob calc
    calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV":energy_range_GeV,
        "ra_rad":ra_rad,
        "dec_rad":dec_rad,
        # "time":time,
    }



    osc_prob = np.zeros((3, len(coszen_range),len(energy_range_GeV)))

    # Simple Earth case
    calculator.set_sme(directional=directional,basis=basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    calculator.set_matter(matter_model)
    osc_prob_loop = calculator.calc_osc_prob(distance_km = baseline, **calc_kw,)
    osc_prob[:,:] = osc_prob_loop.T       # Transpose and select energy node






    #
    # Plot oscillation vs energy
    #
    P_nu_osc_title_dict = {
        0 : r"$P(\nu_{\mu} \to \nu_e)$",
        1 : r"$P(\nu_{\mu} \to \nu_{\mu})$",
        2 : r"$P(\nu_{\mu} \to \nu_{\tau})$",
    }

    fig, ax = plt.subplots(1,3, figsize=(16, 4.5), sharey=False)
    
    for i in range(3):
        ax[i].imshow(osc_prob[i], origin="lower", extent=[energy_range_GeV[0], energy_range_GeV[-1], coszen_range[0], coszen_range[-1]], aspect="auto", cmap="RdPu", vmin=0, vmax=1)
        ax[i].set_xlabel("Energy [GeV]", fontsize=16)
        ax[i].set_ylabel(r"$cos(\theta_{zenith})$", fontsize=16)
        ax[i].set_title(P_nu_osc_title_dict[i], fontsize=16)
        #fix the tick labels
        latex_labels = [r"$10^{}$".format(range(0,6)[i]) for i in range(6)]
        ax[i].set_xticks(ticks=[0,2e4,4e4,6e4,8e4,1e5],labels=latex_labels)
        ax[i].axvline(x=8e4, color="dodgerblue", linestyle="--", lw=2)



        cbar = fig.colorbar(ax[i].images[0], ax=ax[i], orientation="vertical", pad=0.1)
        # cbar.set_label("Oscillation probability", fontsize=16)
        cbar.ax.tick_params(labelsize=14)


    fig.tight_layout()






    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
