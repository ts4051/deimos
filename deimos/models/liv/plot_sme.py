'''
Plo neutrino oscillations with SME parameters activated

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


    #
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1., 1e4, num=1000)
    coszen = -1.


    #
    # Loop over cases
    #

    # Define some physics cases to test
    cases = [ 
        (1e-14, 0.), # a_eV, c
        (0., 1e-26),
    ]

    for a_eV, c in cases :

        # Report
        print("")
        print("Model :")
        print(" a_eV = %s" % a_eV)
        print(" c = %s" % c)


        #
        # Plot oscillation vs energy
        #

        # Calc osc probs and plot, without new physics
        calculator.set_std_osc()
        fig, ax, _, = calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, xscale="log", color="black", label="Standard osc", title=r"coszen = %0.3g"%coszen)

        # Calc osc probs and plot, with SME
        calculator.set_sme(a_eV=a_eV, c=c)
        calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, xscale="log", color="orange", label=r"SME ($a = %0.3g$ eV, $c = %0.3g$)"%(a_eV, c), linestyle="--", fig=fig, ax=ax)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
