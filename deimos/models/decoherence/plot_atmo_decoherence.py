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
    # Create model
    #

    # Create calculator
    calculator = OscCalculator(
        tool="deimos",
        atmospheric=True,
        num_neutrinos=3,
    )

    # Use vacuum
    calculator.set_matter("vacuum")

    # Define decoherence matrix
    gamma_eV = 1e-14
    D_matrix_eV = np.diag([ 0., gamma_eV, gamma_eV, gamma_eV, gamma_eV, gamma_eV, gamma_eV, gamma_eV, gamma_eV ])
    gamma_n = 1 # Energy-dependence
    gamma_E0_eV = 1.e9, # Reference energy, 1 GeV
    print(D_matrix_eV)


    #
    # Plot oscillation vs coszen
    #

    # Define neutrino
    initial_flavor = 1 # muon neutrino
    E_GeV = 100.
    coszen = np.linspace(-1., +1., num=100)

    # Calc osc probs and plot, without decoherence
    calculator.set_std_osc()
    fig, ax, _, = calculator.plot_osc_prob_vs_distance(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, color="black", label="Standard osc")

    # Calc osc probs and plot, with decoherence
    calculator.set_decoherence_D_matrix(D_matrix_eV=D_matrix_eV, n=gamma_n, E0_eV=gamma_E0_eV)
    calculator.plot_osc_prob_vs_distance(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, color="orange", label="Decoherence", linestyle="--", fig=fig, ax=ax)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
