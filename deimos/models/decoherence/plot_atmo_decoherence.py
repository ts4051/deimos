'''
Plot neutirno decoherence in atmopsheric neutrinos

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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Num scan point")
    args = parser.parse_args()


    #
    # Create model
    #

    # Create calculator
    calculator = OscCalculator(
        solver=args.solver,
        atmospheric=True,
    )

    # Use vacuum
    calculator.set_matter("vacuum") #TODO option to add matter

    # Define decoherence matrix
    gamma0_eV = 1e-20
    D_matrix_eV = np.diag([ 0., gamma0_eV, gamma0_eV, gamma0_eV, gamma0_eV, gamma0_eV, gamma0_eV, gamma0_eV, gamma0_eV ])
    gamma_n = 2 # Energy-dependence
    gamma_E0_eV = 1.e9, # Reference energy, 1 GeV

    # Report
    print("")
    print("Model :")
    print(" n = %s" % gamma_n)
    print(" E0 = %s GeV" % gamma_E0_eV)
    print(" gamma0 = %s GeV" % gamma0_eV)
    print(" D matrix [GeV] = \n%s" % D_matrix_eV)
    print("")


    #
    # Plot oscillation vs coszen
    #

    # Define neutrino
    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = 1e4
    coszen = np.linspace(-1., +1., num=args.num_points)

    # Calc osc probs and plot, without decoherence
    calculator.set_std_osc()
    fig, ax, _, = calculator.plot_osc_prob_vs_distance(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, color="black", label="Standard osc", title=r"$E$ = %i TeV"%(E_GeV/1e3))

    # Calc osc probs and plot, with decoherence
    calculator.set_decoherence_D_matrix(D_matrix_eV=D_matrix_eV, n=gamma_n, E0_eV=gamma_E0_eV)
    calculator.plot_osc_prob_vs_distance(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, color="orange", label="Decoherence", linestyle="--", fig=fig, ax=ax)


    #
    # Plot oscillogram
    #

    # Need a grid in atmo. space
    coszen = np.linspace(-1., +1., num=args.num_points)
    E_GeV = np.geomspace(1., 1e5, num=args.num_points)

    # Define final flavor (doesn't scan over all)
    final_flavor = initial_flavor # muon neutrino survival

    # Calc osc probs and plot, without decoherence
    calculator.set_std_osc()
    calculator.plot_oscillogram(initial_flavor=initial_flavor, final_flavor=final_flavor, nubar=nubar, energy_GeV=E_GeV, coszen=coszen, title="Standard osc")

    # Calc osc probs and plot, with decoherence
    calculator.set_decoherence_D_matrix(D_matrix_eV=D_matrix_eV, n=gamma_n, E0_eV=gamma_E0_eV)
    calculator.plot_oscillogram(initial_flavor=initial_flavor, final_flavor=final_flavor, nubar=nubar, energy_GeV=E_GeV, coszen=coszen, title="Decoherence")


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
