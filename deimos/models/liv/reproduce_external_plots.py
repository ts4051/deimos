'''
Reproduce plots from external papers, using our implementation

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *

#
# Plotting functions
#

def reproduce_1410_4267(solver, num_points) :
    '''
    Reproduce plots from arXiv:1410.4267 - SuperK, isotropic LIV

    Cannot directly reproduce all their plots as as some show changes in SuperK event rates rather than 
    oscillation probabilities, but can look to reproduce similar effects at least (e.gh. same periodicity, etc)
    '''


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
        **kw
    )

    # Use vacuum
    calculator.set_matter("vacuum")

    #
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1., 1e4, num=args.num_points)
    coszen = -1.




    #
    # Define physics cases
    #

    sme_basis = "mass"

    null_operator = np.zeros((calculator.num_neutrinos, calculator.num_neutrinos))

    cases = collections.OrderedDict()
    cases[r"$a^{(3)}$ [eV]"] = ( np.diag([0., 0., 1e-14]), null_operator) # (a, c) 
    cases[r"$c^{(4)}$"] = (null_operator, np.diag([0., 0., 1e-26]) )


    #
    # Loop over cases
    #

    for case_label, (a_eV, c) in cases.items() :

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
        calculator.set_sme(directional=False, a_eV=a_eV, c=c)
        calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="orange", label=r"SME : %s"%(case_label), linestyle="--", fig=fig, ax=ax)





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
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1., 1e4, num=args.num_points)
    coszen = -1.


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=False,
        **kw
    )

    # Use vacuum
    calculator.set_matter("vacuum")


    #
    # Define physics cases
    #

    sme_basis = "mass"

    null_operator = np.zeros((calculator.num_neutrinos, calculator.num_neutrinos))

    cases = collections.OrderedDict()
    cases[r"$a^{(3)}$ [eV]"] = ( np.diag([0., 0., 1e-14]), null_operator) # (a, c) 
    cases[r"$c^{(4)}$"] = (null_operator, np.diag([0., 0., 1e-26]) )


    #
    # Loop over cases
    #

    for case_label, (a_eV, c) in cases.items() :

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
        calculator.set_sme(directional=False, a_eV=a_eV, c=c)
        calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="orange", label=r"SME : %s"%(case_label), linestyle="--", fig=fig, ax=ax)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
