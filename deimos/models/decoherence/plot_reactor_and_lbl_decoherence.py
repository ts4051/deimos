'''
Plots of decoherence in reactors and LBL experiments

Used for the decoherence study with Christoph and Valentina

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *


#
# Globals
#

E0_eV = 1.e9

MODELS_COLORS = collections.OrderedDict()

# Models from https://arxiv.org/pdf/2007.00068
MODELS_COLORS["randomize_phase"] = "red"
MODELS_COLORS["randomize_state"] = "blue"
MODELS_COLORS["neutrino_loss"] = "green"

# Models from https://arxiv.org/abs/2306.14699
# MODELS_COLORS["A"] = "red"
# MODELS_COLORS["B"] = "dodgerblue"
# MODELS_COLORS["C"] = "lightgreen"
# MODELS_COLORS["D"] = "orange"
# MODELS_COLORS["E"] = "purple"
# MODELS_COLORS["F"] = "magenta"
# MODELS_COLORS["G"] = "pink"

MODELS = list(MODELS_COLORS.keys())


#
# Plot functions
#

def plot_models(solver, num_points=1000) :
    '''
    Plot a comparison of the various decoherence models tested

    Show cases for both reactors and LBL experiments
    '''


    #
    # Init calc
    #

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=False,
    )

    # Use vacuum
    calculator.set_matter("vacuum")



    #
    # Define experiments
    #

    experiments = collections.OrderedDict()
    
    experiments["NOvA"] = {
        "initial_flavor" : 1,
        "final_flavor" : 1,
        "nubar" : False,
        "L_km" : NOvA_BASELINE_km,
        "E_GeV" : np.linspace(0.5, 5., num=num_points),
        "gamma0_eV" : 1e-22 * 1e9,
        "n" : 0.,
        "ylim" : [0., 1.], 
    }

    experiments["Daya Bay"] = {
        "initial_flavor" : 0,
        "final_flavor" : 0,
        "nubar" : True,
        "L_km" : 1.7, # Furthest detector
        "E_GeV" : np.linspace(1e-3, 10.e-3, num=num_points),
        "gamma0_eV" : 1e-20 * 1e9,
        "n" : 0.,
        "ylim" : [0.8, 1.], 
    }

    #
    # Loop over experiments
    #

    for experiment_key, experiment_def in experiments.items() :

        print("\n%s..." % experiment_key)

        #
        # Plot std osc
        #

        # Set std osc
        calculator.set_std_osc()

        # Plot
        fig, ax, _, = calculator.plot_osc_prob_vs_energy(
            initial_flavor=experiment_def["initial_flavor"], 
            final_flavor=experiment_def["final_flavor"], 
            nubar=experiment_def["nubar"],
            energy_GeV=experiment_def["E_GeV"], 
            distance_km=experiment_def["L_km"], 
            color="black", 
            label="Standard Osc.",
            title=experiment_key,
            ylim=experiment_def["ylim"],
        )


        #
        # Loop over models
        #

        # Loop
        for model_name in MODELS :

            print("\nModel %s..." % model_name)

            # Set model
            calculator.set_decoherence_model(
                model_name=model_name,
                gamma0_eV=experiment_def["gamma0_eV"],
                n=experiment_def["n"],
                E0_eV=E0_eV,
            )

            # Plot (using same fig)
            calculator.plot_osc_prob_vs_energy(
                fig=fig,
                ax=ax,
                initial_flavor=experiment_def["initial_flavor"], 
                final_flavor=experiment_def["final_flavor"], 
                nubar=experiment_def["nubar"],
                energy_GeV=experiment_def["E_GeV"], 
                distance_km=experiment_def["L_km"], 
                color=MODELS_COLORS[model_name], 
                label=model_name.replace("_", " ").title(),
                ylim=experiment_def["ylim"],
            )


#
# Main
#

if __name__ == "__main__" :


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Num scan point")
    args = parser.parse_args()

    plot_models(solver=args.solver, num_points=args.num_points)

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
