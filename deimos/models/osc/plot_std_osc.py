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
    parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Number of points in plots")
    args = parser.parse_args()


    #
    # Create model
    #

    # Tool specific configuration
    kw = {}
    if args.solver == "nusquids" :
        kw["energy_nodes_GeV"] = np.geomspace(0.1, 1000., num=args.num_points) # For nuSQuIDS case, need to specify energy nodes covering full space 

    # Create calculator
    calculator = OscCalculator(
        solver=args.solver,
        atmospheric=False,
        flavors=["e", "mu", "tau"],
        **kw
    )

    # Use vacuum
    calculator.set_matter("vacuum")


    #
    # Plot NOvA
    #

    print("\nPlot NOvA...")

    if calculator.num_neutrinos == 2 :
        initial_flavor, final_flavor, nubar = 0, 0, False
    else :
        initial_flavor, final_flavor, nubar = 1, 1, False

    fig, ax, osc_probs = calculator.plot_osc_prob_vs_energy(
        initial_flavor=initial_flavor, 
        final_flavor=final_flavor, 
        nubar=nubar,
        energy_GeV=np.linspace(0.5, 10., num=args.num_points), # Does not like E=0
        distance_km=810., 
        color="black", 
        label="Standard osc",
        title="NOvA",
    )



    #
    # Plot DeepCore
    #

    print("\nPlot DeepCore...")

    if calculator.num_neutrinos == 2 :
        initial_flavor, final_flavor, nubar = 0, 1, False
    else :
        initial_flavor, final_flavor, nubar = 1, 2, False

    fig, ax, osc_probs = calculator.plot_osc_prob_vs_energy(
        initial_flavor=initial_flavor, 
        final_flavor=final_flavor, 
        nubar=nubar,
        energy_GeV=np.geomspace(1., 200., num=args.num_points), 
        distance_km=EARTH_DIAMETER_km, # coszen = -1 
        color="black", 
        label="Standard osc",
        title="DeepCore",
        xscale="log",
    )

    fig, ax, osc_probs = calculator.plot_osc_prob_vs_distance(
        initial_flavor=initial_flavor, 
        final_flavor=final_flavor, 
        nubar=nubar,
        energy_GeV=25., 
        distance_km=np.linspace(0, EARTH_DIAMETER_km, num=args.num_points), # -> coszen = -1 
        color="black", 
        label="Standard osc",
        title="DeepCore",
    )



    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
