'''
Plot atmospheric neutrino oscillations

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *



#
# Main
#

if __name__ == "__main__" :

    #
    # Steering
    #

    # Get some as args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=250, help="Num scan point")
    parser.add_argument("-m", "--matter", type=str, required=False, default="vacuum", help="Matter")
    args = parser.parse_args()

    # Define neutrino
    initial_flavor, nubar = 1, False # muon neutrino
    final_flavor = initial_flavor # muon neutrino survival

    # Need a grid in atmo. space
    coszen = np.linspace(-1., +1., num=args.num_points)
    E_GeV = np.geomspace(1., 1e5, num=args.num_points)


    #
    # Create model
    #

    # Tool specific configuration
    kw = {}
    if args.solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["interactions"] = True

    # Create calculator
    calculator = OscCalculator(
        solver=args.solver,
        atmospheric=True,
        **kw
    )

    # Use vacuum
    calculator.set_matter(args.matter)


    #
    # Plot oscillogram
    #

    # Calc osc probs and plot
    calculator.plot_oscillogram(initial_flavor=initial_flavor, final_flavor=final_flavor, nubar=nubar, energy_GeV=E_GeV, coszen=coszen)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
