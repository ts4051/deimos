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
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1., 1e4, num=1000)
    coszen = -1.


    #
    # Create model
    #

    # Choose solver
    solver = "nusquids" # deimos nusquids

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = np.geomspace(E_GeV[0], E_GeV[-1], num=100)

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
    # Loop over cases
    #

    # Define some physics cases to test
    cases = collections.OrderedDict()
    cases[r"$a^{(3)}$ [eV]"] = (1e-14, 0) # (coefficient value, energy index n) 
    cases[r"$c^{(4)}$"] = (1e-26, 1)

    for case_label, (cft, n) in cases.items() :

        # Report
        print("")
        print("Model : %s" % case_label)


        #
        # Plot oscillation vs energy
        #

        # Calc osc probs and plot, without new physics
        calculator.set_std_osc()
        fig, ax, _, = calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="black", label="Standard osc", title=r"coszen = %0.3g"%coszen)

        # Calc osc probs and plot, with SME
        calculator.set_sme(cft=cft, n=n)
        calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="orange", label=r"SME : %s = %0.3g"%(case_label, cft), linestyle="--", fig=fig, ax=ax)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
