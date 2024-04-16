'''
Plo neutrino oscillations with sidereal SME parameters activated

Tom Stuttard
'''

import sys, os, collections, datetime
from astropy.time import Time

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *


#
# Main
#

if __name__ == "__main__" :

    #
    # Define basic system
    #

    initial_flavor, final_flavor, nubar = 1, 1, False # numu survival

    ref_E_GeV = 1e3
    ref_time = datetime.datetime(2021, 1, 1, 0, 0, 0, 0) # Midnight, Jan 1st 2021
    ref_ra_deg, ref_dec_deg = 30., 45. 

    E_GeV_scan = np.geomspace(1e2, 1e5, num=100)

    detector_name = "IceCube" # IceCube DUNE ARCA


    #
    # Create model
    #

    # Choose solver
    solver = "nusquids" #"deimos" # deimos nusquids

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        # kw["energy_nodes_GeV"] = np.array([0.5*E_GeV, E_GeV, 1.5*E_GeV])
        kw["energy_nodes_GeV"] = E_GeV_scan
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(
        tool=solver,
        atmospheric=True,
        **kw
    )

    # Use vacuum
    calculator.set_matter("vacuum")

    # Define detector position
    calculator.set_detector(detector_name)



    #
    # Define physics cases
    #

    cases = collections.OrderedDict()

    a_magnitude_eV = 1e-13 # Overall strength of a component
    # c_magnitude = 1e-26 # Overall strength of c component
    flavor_structure = np.array([0., 0., 1.])    #TODO support off-diagonal
    direction_structure = np.array([1., 0., 0.]) # Orient field in x direction     TODO define by RA/dec instead
    a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in direction_structure ])##
    sme_basis = "mass"


    #
    # Plot oscillations vs RA
    #

    # Define neutrino
    ra_deg_values = np.linspace(0., 360., num=100) # Scan
    dec_deg = ref_dec_deg
    dec_deg_values = np.full_like(ra_deg_values, dec_deg)
    time = ref_time
    E_GeV = ref_E_GeV

    # Units
    ra_rad_values = np.deg2rad(ra_deg_values)
    dec_rad_values = np.deg2rad(dec_deg_values)

    # Define common args to osc prob calc
    common_calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV" : E_GeV,
        "ra_rad" : ra_rad_values,
        "dec_rad" : dec_rad_values,
        "time" : time,
    }

    # Get std osc probs
    P_std, _, _ = calculator.calc_osc_prob_sme(std_osc=True, **common_calc_kw)

    # Get LIV osc probs
    P_sme, coszen_values, azimuth_values = calculator.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, **common_calc_kw)

    # Selected chosen final flavor
    P_std = P_std[...,final_flavor]
    P_sme = P_sme[...,final_flavor]

    # Make fig
    fig, ax = plt.subplots( figsize=(6, 4) )

    fig.suptitle( r"$E$ = %0.3g GeV // $\delta$ %0.3g deg // %s" % (E_GeV, dec_deg, time), fontsize=12 )
    
    # Plot osc probs
    ax.plot(ra_deg_values, P_std, color="black", linestyle="-", label="Std. osc.")
    ax.plot(ra_deg_values, P_sme, color="orange", linestyle="--", label="SME")

    #TODO add second x axis with coszen 

    # Plot steering
    P_label = "$%s$" % calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar)

    # Formatting
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel(P_label)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(ra_deg_values[0], ra_deg_values[-1])
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.tight_layout()



    #
    # Plot oscillations vs neutrino direction from detector's perspective, over the course of a day
    #

    # Define neutrino
    ra_deg, dec_deg = ref_ra_deg, ref_dec_deg
    E_GeV = ref_E_GeV

    # Time scan
    hr_values = np.linspace(0., 24., num=48) # TODO sidereal is <24 hrs
    time_values = [ ref_time + datetime.timedelta(hours=hr)  for hr in hr_values ]

    # Define common args to osc prob calc
    common_calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV" : E_GeV,
        "ra_rad" : np.deg2rad(ra_deg),
        "dec_rad" : np.deg2rad(dec_deg),
        "time" : time_values,
    }

    # Get std osc probs
    P_std, _, _ = calculator.calc_osc_prob_sme(std_osc=True, **common_calc_kw)

    # Get LIV osc probs
    P_sme, coszen_values, azimuth_values = calculator.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, **common_calc_kw)

    # Selected chosen final flavor
    P_std = P_std[...,final_flavor]
    P_sme = P_sme[...,final_flavor]

    # Make fig
    fig, ax = plt.subplots( nrows=2, figsize=(6, 6) )

    fig.suptitle(detector_name)
    # fig.suptitle( r"$E$ = %0.3g GeV // $\delta$ = %0.3g deg // %s" % (detector_name, E_GeV, dec_deg, date), fontsize=12 )  #TODO text box
    
    # Plot osc probs
    ax[0].plot(hr_values, P_std, color="black", linestyle="-", label="Std. osc.")
    ax[0].plot(hr_values, P_sme, color="orange", linestyle="--", label="SME")

    # Plot coszen
    ax[1].plot(hr_values, coszen_values, color="blue", linestyle="-")

    #TODO add second x axis with coszen 

    # Formatting
    for this_ax in ax :
        this_ax.set_xlabel("t [hr]")
        this_ax.set_xlim(hr_values[0], hr_values[-1])
        this_ax.grid(True)
    ax[0].set_ylabel(P_label)
    ax[0].set_ylim(-0.01, 1.01)
    ax[0].legend(fontsize=12)
    ax[1].set_ylabel( "coszen" )
    ax[1].set_ylim(-1.01, 1.01)
    fig.tight_layout()



    #
    # Plot oscillations vs energy
    #

    # Define neutrino
    ra_deg, dec_deg = ref_ra_deg, ref_dec_deg
    time = ref_time

    # Define common args to osc prob calc
    common_calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV" : E_GeV_scan,
        "ra_rad" : np.deg2rad(ra_deg),
        "dec_rad" : np.deg2rad(dec_deg),
        "time" : time,
    }

    # Get std osc probs
    P_std, _, _ = calculator.calc_osc_prob_sme(std_osc=True, **common_calc_kw)

    # Get LIV osc probs
    P_sme, coszen_values, azimuth_values = calculator.calc_osc_prob_sme(basis=sme_basis, a_eV=a_eV, **common_calc_kw)

    # Selected chosen final flavor
    P_std = P_std[...,final_flavor]
    P_sme = P_sme[...,final_flavor]

    # Make fig
    fig, ax = plt.subplots( figsize=(6, 4) )

    fig.suptitle(detector_name)
    # fig.suptitle( r"$E$ = %0.3g GeV // $\delta$ = %0.3g deg // %s" % (detector_name, E_GeV, dec_deg, date), fontsize=12 )  #TODO text box
    
    # Plot osc probs
    ax.plot(E_GeV_scan, P_std, color="black", linestyle="-", label="Std. osc.")
    ax.plot(E_GeV_scan, P_sme, color="orange", linestyle="--", label="SME")

    # Formatting
    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_xlim(E_GeV_scan[0], E_GeV_scan[-1])
    ax.set_xscale("log")
    ax.grid(True)
    ax.set_ylabel(P_label)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(fontsize=12)
    fig.tight_layout()


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
