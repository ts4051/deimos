'''
Compare different oscillation solvers (wrapped within the OscCalculator class) to see if they agree
Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *



def compare_osc_solvers() :

    #TODO matter, atmospheric


    #
    # Loop over solvers
    #

    # Init figures
    fig_nova, ax_nova = None, None
    fig_dc, ax_dc = None, None

    # Loop over solers
    for solver, color, linestyle in zip(["deimos", "nusquids", "prob3"], ["blue", "red", "orange"], ["-", "--", ":"]) :

        print("\n\n%s..." % solver)

        #
        # Create model
        #

        # For nuSQuIDS case, need to specify energy nodes covering full space
        kw = {}
        if solver == "nusquids" :
            kw["energy_nodes_GeV"] = np.geomspace(0.1, 1000., num=1000)

        # Create calculator
        calculator = OscCalculator(
            tool=solver,
            atmospheric=False,
            num_neutrinos=3,
            **kw
        )

        # Use vacuum
        calculator.set_matter("vacuum")


        #
        # Plot NOvA
        #

        print("\nPlot NOvA...")

        fig_nova, ax_nova, _ = calculator.plot_osc_prob_vs_energy(
            initial_flavor=1, 
            final_flavor=1, 
            nubar=False,
            energy_GeV=np.linspace(0.5, 10., num=500), # Does not like E=0
            distance_km=810., 
            color=color, 
            linestyle=linestyle,
            label=solver,
            title="NOvA",
            fig=fig_nova,
            ax=ax_nova,
        )



        #
        # Plot DeepCore
        #

        print("\nPlot DeepCore...")

        fig_dc, ax_dc, _ = calculator.plot_osc_prob_vs_energy(
            initial_flavor=1, 
            final_flavor=2, 
            nubar=False,
            energy_GeV=np.geomspace(1., 200., num=500), 
            distance_km=EARTH_DIAMETER_km, # coszen = -1 
            color=color, 
            label=solver,
            linestyle=linestyle,
            title="DeepCore",
            xscale="log",
            fig=fig_dc,
            ax=ax_dc,
        )



#
# Main
#

if __name__ == "__main__" :

    # Plot 
    compare_osc_solvers()

    # Svae figs
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
