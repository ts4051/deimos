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

SOLVER = "deimos"

E0_eV = 1.e9

MODELS_COLORS = collections.OrderedDict()
MODELS_COLORS["A"] = "red"
MODELS_COLORS["B"] = "dodgerblue"
MODELS_COLORS["C"] = "lightgreen"
MODELS_COLORS["D"] = "orange"
MODELS_COLORS["E"] = "purple"
MODELS_COLORS["F"] = "magenta"
MODELS_COLORS["G"] = "pink"

MODELS = list(MODELS_COLORS.keys())

NUM_SCAN_POINTS = 1000


#
# Helper functions
#


def get_model_D_matrix(name, gamma) :
    '''
    Return the D matrix for a given model
    '''

    #
    # Get model def
    #

    if name == "A" :
        gamma21 = gamma
        gamma31 = gamma
        gamma32 = gamma

    elif name == "B" :
        gamma21 = gamma
        gamma31 = gamma
        gamma32 = 0.

    elif name == "C" :
        gamma21 = gamma
        gamma31 = 0.
        gamma32 = gamma

    elif name == "D" :
        gamma21 = 0.
        gamma31 = gamma
        gamma32 = gamma

    elif name == "E" :
        gamma21 = gamma
        gamma31 = 0.
        gamma32 = 0.

    elif name == "F" :
        gamma21 = 0.
        gamma31 = gamma
        gamma32 = 0.

    elif name == "G" :
        gamma21 = 0.
        gamma31 = 0.
        gamma32 = gamma

    else :
        raise Exception("Unknown model")


    #
    # Form the D matrix
    #

    D = np.diag([gamma21, gamma21, 0., gamma31, gamma31, gamma32, gamma32, 0.])

    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????
    #TODO Is this the correct format? - round trip test wth get_decoherence_operator_nxn_basis????

    return D


#
# Plot functions
#

def plot_models() :
    '''
    Plot a comparison of the various decoherence models tested

    Show cases for both reactors and LBL experiments
    '''


    #
    # Init calc
    #

    # Create calculator
    calculator = OscCalculator(
        tool=SOLVER,
        atmospheric=False,
        num_neutrinos=3,
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
        "E_GeV" : np.linspace(0.5, 5., num=NUM_SCAN_POINTS),
        "gamma_eV" : 1e-22 * 1e9,
        "n" : 0.,
        "ylim" : [0., 1.], 
    }

    experiments["Daya Bay"] = {
        "initial_flavor" : 0,
        "final_flavor" : 0,
        "nubar" : True,
        "L_km" : 1.7, # Furthest detector
        "E_GeV" : np.linspace(1e-3, 10.e-3, num=NUM_SCAN_POINTS),
        "gamma_eV" : 1e-20 * 1e9,
        "n" : 0.,
        "ylim" : [0.85, 1.], 
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
            label="Standard osc",
            title=experiment_key,
            ylim=experiment_def["ylim"],
        )


        #
        # Loop over models
        #

        # Loop
        for model_name in MODELS :

            print("\nModel %s..." % model_name)

            # Get model D matrix
            D_matrix_eV = get_model_D_matrix(name=model_name, gamma=experiment_def["gamma_eV"])

            # Set model
            calculator.set_decoherence_D_matrix(
                D_matrix_eV=D_matrix_eV,
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
                label=model_name,
                ylim=experiment_def["ylim"],
            )



    # def set_decoherence_D_matrix(self,
    #     D_matrix_eV,
    #     n, # energy-dependence
    #     E0_eV,
    # ) :


def plot_unphysical_models_issue() :
    '''
    There seems to be an issue with unphysical oscillation probs (-ve) when in cases where one of gamma_ij = 0 and the other two gammsa_ij are equal and strong

    Investigating this here
    '''

    pass



#
# Main
#

if __name__ == "__main__" :

    #
    # Run plotters
    #

    plot_models()

    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
