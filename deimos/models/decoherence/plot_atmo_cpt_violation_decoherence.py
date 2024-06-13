'''
A more detailed dive into CPT-V decoherence terms in atmopsheric neutrinos.

Tom Stuttard, Christoph Ternes
'''

import sys, os, collections, copy

import matplotlib.pyplot as plt
import numpy as np

from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.constants import *
# from deimos.model.decoherence_operators import oscillation_averaged_transition_probability, convert_gamma_eV_to_gamma_inv_m, convert_gamma_inv_m_to_gamma_eV, convert_gamma_eV_to_gamma_0_eV, convert_gamma0_to_zeta_planck
from deimos.models.decoherence.nuVBH_model import *
from deimos.utils.plotting import *


#
# Globals
#

# Choose a reference gamma value to use
# REF_GAMMA_eV = convert_gamma_inv_m_to_gamma_eV(REF_COHERENCE_LENGTH_m)


#
# Helper functions
#

def format_oscillogram_ax(ax, E_GeV, coszen) :
    ax.grid(False)
    ax.set_xlabel(ENERGY_LABEL)
    ax.set_ylabel(COSZEN_LABEL)
    ax.set_xscale("log")
    ax.set_xlim(E_GeV[0],E_GeV[-1])
    ax.set_ylim(coszen[0],coszen[-1])


#
# Plotting functions
#

def plot_different_models(
    solver,
    num_points=10,
) :

    #
    # Create calculator
    #

    # Define parameter space
    coszen = np.linspace(-1., +1., num=num_points)
    E_GeV = np.logspace(0., 5., num=num_points)

    # Handle solver-specific stuff
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV # Put nodes on the scan points (faster)
        kw["coszen_nodes"] = coszen # Put nodes on the scan points (faster)
        kw["interactions"] = True
        kw["nusquids_variant"] = "decoherence"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
        num_neutrinos=3,
        **kw
    )

    # Define system
    calculator.set_matter("vacuum") #TODO Earth model not yet implemented


    #
    # Decoherence matrix cases
    #

    # Using the cases from arXiv 1811.04982 Fig. 1

    n = 0

    E0_eV = 1e9

    gamma_eV = 1e-23 * 1e9
    # gamma_eV = REF_GAMMA_eV

    beta28_eV = gamma_eV / np.sqrt(3.)
    beta12_eV = gamma_eV / 3.
    beta56_eV = gamma_eV / np.sqrt(3.)
    beta47_eV = gamma_eV / np.sqrt(3.)

    diag_matrix = np.diag([0] + [gamma_eV]*8) # State selection case

    cases = collections.OrderedDict()

    D = diag_matrix.copy()
    cases[r"Diagonal"] = {
        "D_matrix_eV" : D,
        "color" : "black",
        "linestyle" : "-",
    }

    D = diag_matrix.copy()
    D[1,2] = beta12_eV
    D[2,1] = beta12_eV
    cases[r"$\beta_{12}$"] = {
        "D_matrix_eV" : D,
        "color" : "blue",
        "linestyle" : "--",
    }

    D = diag_matrix.copy()
    D[2,8] = beta28_eV
    D[8,2] = beta28_eV
    cases[r"$\beta_{28}$"] = {
        "D_matrix_eV" : D,
        "color" : "red",
        "linestyle" : "--",
    }

    D = diag_matrix.copy()
    D[4,7] = beta47_eV
    D[7,4] = beta47_eV
    cases[r"$\beta_{47}$"] = {
        "D_matrix_eV" : D,
        "color" : "magenta",
        "linestyle" : "--",
    }

    D = diag_matrix.copy()
    D[5,6] = beta56_eV
    D[6,5] = beta56_eV
    cases[r"$\beta_{56}$"] = {
        "D_matrix_eV" : D,
        "color" : "orange",
        "linestyle" : "--",
    }


    #
    # Plot osc prob
    #

    # Choose channels
    initial_flavor_values = [ 0, 1 ] # e, mu
    final_flavor_values = [ 0, 1, 2 ] # e, mu, tau

    # Loop over inital flavor
    for i_i, initial_flavor in enumerate(initial_flavor_values) :

        # Define calc_osc_prob kwargs once only
        calc_osc_prob_kw = {
            "initial_flavor" : initial_flavor,
            "energy_GeV" : E_GeV,
            "coszen" : coszen,
        }

        # Loop over final flavor
        for i_f, final_flavor in enumerate(final_flavor_values) :
    
            # Plot steering
            nu_transition_prob_tex = calculator.get_transition_prob_tex(initial_flavor, final_flavor, False)
            nubar_transition_prob_tex = calculator.get_transition_prob_tex(initial_flavor, final_flavor, True)
            Acpt_label = r"$A_{CPT}$"
            Acpt_range = [-0.1, +0.1]

            # Add heading page
            add_heading_page(r"$%s$" % nu_transition_prob_tex)

            # Create 1D figure
            fig_1d, ax_1d = plt.subplots( ncols=3, nrows=2, figsize=(12, 8) )
            # fig_2d.suptitle(r"$%s$" % nu_transition_prob_tex)


            #
            # Standard oscillations
            #

            # Set standard oscillations (disable any non-standard effects)
            calculator.set_std_osc()

            # Calculate std osc probs
            osc_probs_std_nu = calculator.calc_osc_prob(nubar=False, **calc_osc_prob_kw)[...,final_flavor]
            osc_probs_std_nubar = calculator.calc_osc_prob(nubar=True, **calc_osc_prob_kw)[...,final_flavor]

            # Calculate asymmetry
            Acpt_std = osc_probs_std_nu - osc_probs_std_nubar

            # Plot 1D oscillations vs energy (for coszen=1)
            ref_idx = 0
            ref_coszen = coszen[ref_idx] #TODO -> title
            plot_kw = {"color":"grey", "linewidth":2, "linestyle":":", "label":"Std osc"}
            ax_1d[0, 0].plot( E_GeV, osc_probs_std_nu[:,ref_idx], **plot_kw )
            ax_1d[0, 1].plot( E_GeV, osc_probs_std_nubar[:,ref_idx], **plot_kw )
            ax_1d[0, 2].plot( E_GeV, Acpt_std[:,ref_idx], **plot_kw )

            # Plot oscillograms
            fig_2d, ax_2d = plt.subplots( ncols=3,figsize=(18, 5) )
            fig_2d.suptitle("Standard oscillations")
            plot_colormap( ax=ax_2d[0], x=E_GeV, y=coszen, z=osc_probs_std_nu, vmin=0., vmax=1., cmap="jet", zlabel=r"$%s$"%nu_transition_prob_tex )
            plot_colormap( ax=ax_2d[1], x=E_GeV, y=coszen, z=osc_probs_std_nubar, vmin=0., vmax=1., cmap="jet", zlabel=r"$%s$"%nubar_transition_prob_tex )
            plot_colormap( ax=ax_2d[2], x=E_GeV, y=coszen, z=Acpt_std, vmin=Acpt_range[0], vmax=Acpt_range[1], cmap="PuOr_r", zlabel=Acpt_label )
            for this_ax in ax_2d.flatten() :
                format_oscillogram_ax(this_ax, E_GeV, coszen)
            fig_2d.tight_layout()


            #
            # Decoherence cases
            #

            # Loop over cases
            for i, (label, case_dict) in enumerate(cases.items()) :

                # Set decoherence model
                calculator.set_decoherence_D_matrix(D_matrix_eV=case_dict["D_matrix_eV"], n=n, E0_eV=E0_eV)

                # Calculate decoh osc probs
                osc_probs_decoh_nu = calculator.calc_osc_prob(nubar=False, **calc_osc_prob_kw)[...,final_flavor]
                osc_probs_decoh_nubar = calculator.calc_osc_prob(nubar=True, **calc_osc_prob_kw)[...,final_flavor]

                # Calculate asymmetry
                Acpt_decoh = osc_probs_decoh_nu - osc_probs_decoh_nubar

                # Calculate difference w.r.t. standard oscillations
                decoh_vs_std_osc_probs_nu = osc_probs_decoh_nu - osc_probs_std_nu
                decoh_vs_std_osc_probs_nubar = osc_probs_decoh_nubar - osc_probs_std_nubar
                decoh_vs_std_Acpt = Acpt_decoh - Acpt_std

                # Plot 1D oscillations vs energy (for coszen=1)
                plot_kw = {"color":case_dict["color"], "linewidth":2, "linestyle":case_dict["linestyle"], "label":label}
                ax_1d[0, 0].plot( E_GeV, osc_probs_decoh_nu[:,ref_idx], **plot_kw )
                ax_1d[0, 1].plot( E_GeV, osc_probs_decoh_nubar[:,ref_idx], **plot_kw )
                ax_1d[0, 2].plot( E_GeV, Acpt_decoh[:,ref_idx], **plot_kw )
                ax_1d[1, 0].plot( E_GeV, decoh_vs_std_osc_probs_nu[:,ref_idx], **plot_kw )
                ax_1d[1, 1].plot( E_GeV, decoh_vs_std_osc_probs_nubar[:,ref_idx], **plot_kw )
                ax_1d[1, 2].plot( E_GeV, decoh_vs_std_Acpt[:,ref_idx], **plot_kw )

                # Plot oscillograms
                fig_2d, ax_2d = plt.subplots( ncols=3, nrows=2, figsize=(18, 10) )
                fig_2d.suptitle(label)
                plot_colormap( ax=ax_2d[0, 0], x=E_GeV, y=coszen, z=osc_probs_decoh_nu, vmin=0., vmax=1., cmap="jet", zlabel=r"$%s$"%nu_transition_prob_tex )
                plot_colormap( ax=ax_2d[0, 1], x=E_GeV, y=coszen, z=osc_probs_decoh_nubar, vmin=0., vmax=1., cmap="jet", zlabel=r"$%s$"%nubar_transition_prob_tex )
                plot_colormap( ax=ax_2d[0, 2], x=E_GeV, y=coszen, z=Acpt_decoh, vmin=Acpt_range[0], vmax=Acpt_range[1], cmap="PuOr_r", zlabel=Acpt_label )
                plot_colormap( ax=ax_2d[1, 0], x=E_GeV, y=coszen, z=decoh_vs_std_osc_probs_nu, vmin=-0., vmax=1., cmap="PuOr_r", zlabel=r"$%s$ (decoh - std)"%nu_transition_prob_tex )
                plot_colormap( ax=ax_2d[1, 1], x=E_GeV, y=coszen, z=decoh_vs_std_osc_probs_nubar, vmin=-0., vmax=1., cmap="PuOr_r", zlabel=r"$%s (decoh - std)$"%nubar_transition_prob_tex )
                plot_colormap( ax=ax_2d[1, 2], x=E_GeV, y=coszen, z=decoh_vs_std_Acpt, vmin=Acpt_range[0], vmax=Acpt_range[1], cmap="PuOr_r", zlabel=r"%s (decoh - std)"%Acpt_label )
                for this_ax in ax_2d.flatten() :
                    format_oscillogram_ax(this_ax, E_GeV, coszen)
                fig_2d.tight_layout()


            # Format 1D figure
            ax_1d[0, 0].set_ylabel(r"$%s$" % nu_transition_prob_tex)
            ax_1d[0, 1].set_ylabel(r"$%s$" % nubar_transition_prob_tex)
            ax_1d[0, 2].set_ylabel(Acpt_label)
            ax_1d[1, 0].set_ylabel(r"$%s$ (decoh - std)" % nu_transition_prob_tex)
            ax_1d[1, 1].set_ylabel(r"$%s$ (decoh - std)" % nubar_transition_prob_tex)
            ax_1d[1, 2].set_ylabel("$%s (decoh - std)" % Acpt_label)
            ax_1d[0, 0].set_ylim(0., 1.)
            ax_1d[0, 1].set_ylim(0., 1.)
            ax_1d[0, 2].set_ylim(Acpt_range)
            ax_1d[1, 0].set_ylim(-1., 1.)
            ax_1d[1, 1].set_ylim(-1, 1.)
            ax_1d[1, 2].set_ylim(Acpt_range)
            for this_ax in ax_1d.flatten() :
                this_ax.grid(True)
                this_ax.set_xlabel(ENERGY_LABEL)
                this_ax.set_xscale("log")
                this_ax.set_xlim(E_GeV[0], E_GeV[-1])
                this_ax.legend(fontsize=8)
            fig_1d.tight_layout()



#
# Main
#

if __name__ == "__main__" :

    #
    # Plotting
    #

    # Steering
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, default="deimos", help="Name of solver")
    parser.add_argument("-n", "--num-points", type=int, default=100, help="Number of points in the solver")
    args = parser.parse_args()

    # Run each plotting function
    plot_different_models(solver=args.solver, num_points=args.num_points)

    # Dump figures
    print("")
    output_file = __file__.split(".py")[0] + "_" + args.solver + ".pdf"
    dump_figures_to_pdf(output_file)
