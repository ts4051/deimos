'''
Comparing decoherence implementations in different solvers

Tom Stuttard
'''

import sys, os, collections

from matplotlib import cm

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *

from deimos.models.decoherence.nuVBH_model import get_vVBH_gamma_eV_from_coherence_length_km, convert_gamma_eV_to_gamma_0_eV



#
# Main
#

if __name__ == "__main__" :


    #
    # Define scenarios
    #

    # Steer what to plot
    debug = False
    include_plots_vs_energy = True # This is slow with DEIMOS

    # Define common settings
    mixing_angles_rad = np.deg2rad( np.array([ 35., 9., 40. ]) ) # Roughly realistic values, except avoiding maximal mixing for theta23 to help differntiate decoherence models
    deltacp_rad = np.deg2rad(90.) #TODO what value?
    mass_splittings_eV2 = np.array([7.5e-5, 2.5e-3])
    E0_eV = 1e9
    initial_flavor = 1
    final_flavor_values = [0, 1, 2]
    nubar_values = [False, True] 

    # Define scenarios
    cases = collections.OrderedDict()
    for energy_label, E_GeV in [ ("GeV", 25.), ("TeV", 1e4) ] : # GeV and TeV scales (e.g. both including std oscillations and not)
        for n in [ 0, 2 ] : # Energy-dependence
            for model_name in ["randomize_phase", "randomize_state", "neutrino_loss"] :
                for matter_label, matter_kw in [ ("Vacuum", {"matter":"vacuum"}), ("Const density", {"matter":"constant", "matter_density_g_per_cm3":50., "electron_fraction":0.5}) ] :
                    cases[f"{energy_label}, {model_name}, n={n}, {matter_label}"] = { "E_GeV":E_GeV, "matter_kw":matter_kw, "model_name":model_name, "n":n  }


    #
    # Loop over decoherence cases
    #

    # Loop over cases
    for i_case, (case_label, case_kw) in enumerate( cases.items() ) :

        print("")
        print(f"{case_label}...")


        #
        # Choose decoherence strength
        #

        # For the chose neutrino energy and decoherence energy-dependence, choose a gamma0 value that gives Earth-scaled 
        # coherence lengths so we can see a significant effect

        gamma_eV = get_vVBH_gamma_eV_from_coherence_length_km(EARTH_DIAMETER_km)
        gamma0_eV = convert_gamma_eV_to_gamma_0_eV(gamma_eV=gamma_eV, E_eV=case_kw["E_GeV"]*1e9, E0_eV=E0_eV, n=n)


        #
        # Loop over energy vs distance plot
        #

        for plot_vs_energy in ([False, True] if include_plots_vs_energy else [False]) :

            print("Plotting vs %s..." % ("energy" if plot_vs_energy else "coszen"))


            #
            # Define scan
            #

            num_scan_points = 50 if debug else 200

            xscale = "linear"

            if plot_vs_energy :

                # Energy scan
                L_km = EARTH_DIAMETER_km
                if case_kw["E_GeV"] > 100. :
                    E_GeV = np.geomspace(1., case_kw["E_GeV"], num=num_scan_points)
                    xscale = "log"
                else :
                    E_GeV = np.linspace(1., case_kw["E_GeV"], num=num_scan_points)

            else :

                # Distance scan
                L_km = np.linspace(0., EARTH_DIAMETER_km)
                E_GeV = case_kw["E_GeV"]



            #
            # Init figure
            #

            nx, ny = len(nubar_values), len(final_flavor_values)
            fig, ax = plt.subplots( ncols=nx, nrows=ny, figsize=(4*nx, 2*ny) )

            fig.suptitle(case_label)



            #
            # Loop over solvers
            #

            solvers = [ "deimos", "nusquids" ]
            # solvers = [ "deimos" ]
            linestyles = [ "-", "--", ":" ]
            color_adjustments = [ 0.7, 1., 1.5]

            for solver, linestyle, color_adjust in zip(solvers, linestyles, color_adjustments) :

                print(f"Plotting using {solver}...")


                #
                # Create model
                #

                # Tool specific configuration
                kw = {}
                if solver == "nusquids" :
                    kw["energy_nodes_GeV"] = E_GeV # Faster to set the nodes at the values we will calc osc probs for (and thus avoid interpolation)
                    kw["interactions"] = False # Not available in other solvers in general
                    kw["nusquids_variant"] = "decoherence"

                # Create calculator
                calculator = OscCalculator(
                    tool=solver,
                    atmospheric=False,
                    mixing_angles_rad=mixing_angles_rad,
                    deltacp_rad=deltacp_rad,
                    mass_splittings_eV2=mass_splittings_eV2,
                    **kw
                )

                # Set matter
                calculator.set_matter(**case_kw["matter_kw"])


                #
                # Plot both decoherence and standard oscillations
                #

                for std_osc, color in zip([True, False], ["grey", "orange"]) :

                    # Set decoherence model, or disable
                    if std_osc :
                        calculator.set_std_osc()
                        label = "Std osc"
                    else :
                        calculator.set_decoherence_model(model_name=case_kw["model_name"], gamma0_eV=gamma0_eV, n=n, E0_eV=E0_eV)
                        label = "Decoherence"


                    #
                    # Plot osc probs
                    #

                    # Loop over nu/nubar
                    for x, nubar in enumerate(nubar_values) :

                        # Calc osc probs
                        osc_probs = calculator.calc_osc_prob(
                            initial_flavor=initial_flavor, 
                            nubar=nubar,
                            energy_GeV=E_GeV,
                            distance_km=L_km,
                        )

                        # Loop over final flavors
                        for y, final_flavor in enumerate(final_flavor_values) :

                            # Plot osc probs
                            x_plot = E_GeV if plot_vs_energy else L_km
                            ax[y,x].plot(
                                x_plot, 
                                osc_probs[:,final_flavor], 
                                label="%s [%s]"%(label, solver), 
                                linestyle=linestyle,   
                                color=adjust_lightness(color, color_adjust),
                                lw=3,
                            )

                            # Format ax
                            ax[y,x].set_xscale(xscale)
                            ax[y,x].set_xlim(x_plot[0], x_plot[-1])
                            # if final_flavor == 0 :
                            #     ax[y,x].set_ylim(0., 0.2)
                            # else :
                            ax[y,x].set_ylim(-0.02, 1.02)
                            ax[y,x].set_ylabel( r"$%s$"%calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )
                            ax[y,x].grid(True)
                            if (y==0) and (x==0) :
                                ax[y,x].legend(ncol=len(solvers), fontsize=6, loc="upper center")


                        # More ax formatting
                        ax[-1,x].set_xlabel( ENERGY_LABEL if plot_vs_energy else DISTANCE_LABEL )

            # Format figure
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.tight_layout()


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
