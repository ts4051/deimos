'''
Make some basic examples plots of standard oscillations

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *


#
# Main
#

if __name__ == "__main__" :

    #
    # Steering
    #

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    args = parser.parse_args()


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if args.solver == "nusquids" :
        kw["energy_nodes_GeV"] = np.geomspace(0.1, 1000., num=1000)

    # Create calculator
    calculator = OscCalculator(
        tool=args.solver,
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

    fig, ax, osc_probs = calculator.plot_osc_prob_vs_energy(
        initial_flavor=1, 
        final_flavor=1, 
        nubar=False,
        energy_GeV=np.linspace(1e-3, 10., num=500), # Does not like E=0
        distance_km=810., 
        color="black", 
        label="Standard osc",
        title="NOvA",
    )



    #
    # Plot DeepCore
    #

    print("\nPlot DeepCore...")

    fig, ax, osc_probs = calculator.plot_osc_prob_vs_energy(
        initial_flavor=1, 
        final_flavor=2, 
        nubar=False,
        energy_GeV=np.geomspace(1., 200., num=500), 
        distance_km=EARTH_DIAMETER_km, # coszen = -1 
        color="black", 
        label="Standard osc",
        title="DeepCore",
        xscale="log",
    )


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
