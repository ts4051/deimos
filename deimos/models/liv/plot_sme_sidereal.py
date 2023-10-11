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
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1., 1e4, num=1000)
    coszen = -1.


    #
    # Create model
    #

    # Choose solver
    solver = "deimos" # deimos nusquids

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = np.geomspace(E_GeV[0], E_GeV[-1], num=100)

    # Create calculator
    calculator = OscCalculator(
        tool=solver,
        atmospheric=True,
        num_neutrinos=3,
        **kw
    )

    # Use vacuum
    calculator.set_matter("vacuum")

    # Define detector position
    detector_name = "DUNE" # IceCube DUNE
    calculator.set_detector(detector_name)


    #
    # Define basic system
    #

    initial_flavor, final_flavor, nubar = 2, 2, False # numu survival

    E_GeV = 1e3


    #
    # Define physics cases
    #

    cases = collections.OrderedDict()

    directional = True
    sme_basis = "mass"

    a_magnitude_eV = 1e-13 # Overall strength of a component
    c_magnitude = 1e-26 # Overall strength of c component

    # flavor_structure = np.array([1., 1., 1.]) # Equally shared between flavors (diagonal)
    flavor_structure = np.array([0., 0., 1.]) # Equally shared between flavors (diagonal)

    if directional :
        direction_structure = np.array([1., 0., 0.]) # Orient field in x direction
        cases["a"] = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in direction_structure ])
        null_operator = np.zeros( (3, calculator.num_neutrinos, calculator.num_neutrinos) )

    else :
        cases["a"] = a_magnitude_eV * np.diag(flavor_structure)
        null_operator = np.zeros( (calculator.num_neutrinos, calculator.num_neutrinos) )


    #
    # Loop over cases
    #


    # Loop over cases
    for case_label, (a_eV) in cases.items() :

        # Report
        print("")
        print("Model : %s" % case_label)


        #
        # Plot oscillations vs RA
        #

        # Define neutrino
        ra_values_deg = np.linspace(0., 360., num=100)
        dec_deg = +30
        time = "July 15, 2020, 14:30"

        # Units
        dec_rad = np.deg2rad(dec_deg)

        # Loop over RA
        P_std, P_liv = [], []
        for ra_deg in ra_values_deg :

            # Units
            ra_rad = np.deg2rad(ra_deg)

            # Get coszen/azimuth from RA/dec
            coszen, _, azimuth_deg = calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg, dec_deg=dec_deg, time=time)

            # Define args to osc prob calc
            calc_kw = {
                "initial_flavor":initial_flavor,
                "nubar" : nubar,
                "energy_GeV":E_GeV,
                "coszen":coszen,
                "ra_rad":ra_rad,
                "dec_rad":dec_rad,
            }

            # Get std osc probs
            calculator.set_std_osc()
            P_std.append( np.squeeze( calculator.calc_osc_prob(**calc_kw)[...,final_flavor] ) ) # Single value

            # Get LIV osc probs
            calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=null_operator, e=null_operator)
            P_liv.append( np.squeeze( calculator.calc_osc_prob(**calc_kw)[...,final_flavor] ) ) # Single value

        # Numpy-ify
        P_std = np.array(P_std)
        P_liv = np.array(P_liv)
        assert ra_values_deg.shape == P_std.shape
        assert ra_values_deg.shape == P_liv.shape

        # Make fig
        fig, ax = plt.subplots( figsize=(6, 4) )

        fig.suptitle( r"$E$ = %0.3g GeV // $\delta$ %0.3g deg // %s" % (E_GeV, dec_deg, time), fontsize=12 )
        
        # Plot osc probs
        ax.plot(ra_values_deg, P_std, color="black", linestyle="-", label="Std. osc.")
        ax.plot(ra_values_deg, P_liv, color="orange", linestyle="--", label=case_label)

        #TODO add second x axis with coszen 

        # Formatting
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel( "$%s$" % calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(ra_values_deg[0], ra_values_deg[-1])
        ax.grid(True)
        ax.legend(fontsize=12)
        fig.tight_layout()


        #
        # Plot oscillations vs neutrino direction from detector's perspective, over the course of a day
        #

        # Choose neutrino direction
        ra_deg, dec_deg = 30., 15. #TODO what to choose (so far this is just random)?

        # Time scan
        # t0 = datetime.datetime.strptime("2020/01/01, 00:00", "%y/%m/%d %H:%M")
        t0 = datetime.datetime(2021, 1, 1, 0, 0, 0, 0) # Midnight, Jan 1st 2021
        hr_values = np.linspace(0., 24., num=100) # TODO sidereal is <24 hrs

        # Loop over times
        P_std, P_liv, coszen_values = [], [], []
        for hr in hr_values :

            # Get coszen/azimuth from RA/dec
            time = t0 + datetime.timedelta(hours=hr)
            coszen, _, azimuth_deg = calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=ra_deg, dec_deg=dec_deg, time=time)
            coszen_values.append(coszen)

            # Convert to baseline
            #TODO (for plotting only)

            # Define args to osc prob calc
            calc_kw = {
                "initial_flavor":initial_flavor,
                "nubar" : nubar,
                "energy_GeV":E_GeV,
                "coszen":coszen,
                "ra_rad":ra_rad,
                "dec_rad":dec_rad,
            }

            # Get std osc probs
            calculator.set_std_osc()
            P_std.append( np.squeeze( calculator.calc_osc_prob(**calc_kw)[...,final_flavor] ) ) # Single value

            # Get LIV osc probs
            calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=null_operator, e=null_operator)
            P_liv.append( np.squeeze( calculator.calc_osc_prob(**calc_kw)[...,final_flavor] ) ) # Single value

        # Numpy-ify
        P_std = np.array(P_std)
        P_liv = np.array(P_liv)
        coszen_values = np.array(coszen_values)
        assert hr_values.shape == P_std.shape
        assert hr_values.shape == P_liv.shape
        assert hr_values.shape == coszen_values.shape

        # Make fig
        fig, ax = plt.subplots( nrows=2, figsize=(6, 8) )

        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time
        #TODO also plot coszen vs time

        fig.suptitle(detector_name)
        # fig.suptitle( r"$E$ = %0.3g GeV // $\delta$ = %0.3g deg // %s" % (E_GeV, dec_deg, date), fontsize=12 )  #TODO text box
        
        # Plot osc probs
        ax[0].plot(hr_values, P_std, color="black", linestyle="-", label="Std. osc.")
        ax[0].plot(hr_values, P_liv, color="orange", linestyle="--", label=case_label)

        # Plot coszen
        ax[1].plot(hr_values, coszen_values, color="blue", linestyle="-")

        #TODO add second x axis with coszen 

        # Formatting
        for this_ax in ax :
            this_ax.set_xlabel("t [hr]")
            this_ax.set_xlim(hr_values[0], hr_values[-1])
            this_ax.grid(True)
        ax[0].set_ylabel( "$%s$" % calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )
        ax[0].set_ylim(-0.01, 1.01)
        ax[0].legend(fontsize=12)
        ax[1].set_ylabel( "coszen" )
        ax[1].set_ylim(-1.01, 1.01)
        fig.tight_layout()



            # print(ra_deg, coszen, azimuth_deg)






        # #
        # # Plot oscillation vs energy
        # #

        # # Calc osc probs and plot, without new physics
        # calculator.set_std_osc()
        # fig, ax, _, = calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="black", label="Standard osc", title=r"coszen = %0.3g"%coszen)

        # # Calc osc probs and plot, with SME
        # calculator.set_sme(cft=cft, n=n)
        # calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="orange", label=r"SME : %s = %0.3g"%(case_label, cft), linestyle="--", fig=fig, ax=ax)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
