'''
Making atmopsheric neutrino decoherence plots for
a decoherence theory paper.

These are "everything in" plots showing all modelled 
physics.

Tom Stuttard
'''

import sys, os, collections, copy

import matplotlib.pyplot as plt
import numpy as np

from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.constants import *
# from deimos.model.decoherence_operators import oscillation_averaged_transition_probability, convert_gamma_eV_to_gamma_inv_m, convert_gamma_inv_m_to_gamma_eV, convert_gamma_eV_to_gamma_0_eV, convert_gamma0_to_zeta_planck
from deimos.models.decoherence.nuVBH_model import *

#
# Globals
#

# Choose a reference gamma value to use
REF_GAMMA_eV = convert_gamma_inv_m_to_gamma_eV(REF_COHERENCE_LENGTH_m)

# Define solver backend
TOOL = "deimos"


#
# Plotting functions
#

def plot_arxiv_1807_07823(
    num_points=1000,
) :
    '''
    Re-produce plots from arXiv 1807.07823
    '''

    print("\n>>> arXiv 1807.07823")

    #
    # Create calculator
    #

    # assert tool != "nusquids", "2-flavor not supported by nuSQuIDS" - TODO I think I added this...

    calculator = OscCalculator(
        tool=TOOL,
        atmospheric=False,
        num_neutrinos=2,
    )

    calculator.set_calc_basis("nxn")
    calculator.set_matter("vacuum")

    n = 0
    E0_eV = 1e9


    #
    # DeepCore case
    #

    # Reproducing Fig 1

    # Params
    sin2_theta23 = 0.51
    theta23_rad = np.arcsin(np.sqrt(sin2_theta23))
    dm32_eV2 = 2.5e-3 #TODO convert to 31?
    gamma_eV = 4e-24 * 1e9
    gamma3_eV = 7.9e-24 * 1e9
    alpha_eV = 3.8e-24 * 1e9
    deltacp = 0.

    # Set them
    calculator.set_mass_splittings(dm32_eV2)
    calculator.set_mixing_angles(theta23_rad, deltacp=deltacp)

    # Decoherence matrix
    D_matrix_eV = np.array([ # eqn 8
        [ 0., 0.,       0.,       0.        ],
        [ 0., gamma_eV, alpha_eV, 0.        ],
        [ 0., alpha_eV, gamma_eV, 0.        ],
        [ 0., 0.,       0.,       gamma3_eV ],
    ])

    # Choose baseline, energy, etc
    L_km = EARTH_DIAMETER_km
    E_GeV = np.linspace(6., 120., num=num_points)
    initial_flavor = 0
    final_flavor = 1
    rho_values = [0, 1]

    # Plot steering
    std_osc_color = "blue"
    decoh_color = "red"
    linestyles = ["-", "--"]
    initial_flavor_tex = r"$\nu_\mu$"
    final_flavor_tex = r"$\nu_\tau$"
    rho_tex = [ r"\nu", r"\bar{\nu}" ]

    # Create the fig
    fig, ax = plt.subplots( figsize=(6, 4) )

    # Std oscillations
    calculator.set_std_osc()
    for rho in rho_values :
        osc_probs = calculator.calc_osc_prob(
            initial_flavor=initial_flavor,
            initial_rho=rho,
            energy_GeV=E_GeV,
            distance_km=L_km,
        )
        label = calculator.get_transition_prob_tex(initial_flavor=initial_flavor, final_flavor=final_flavor, initial_flavor_nubar=(rho==1), final_flavor_nubar=(rho==1))
        ax.plot( E_GeV, osc_probs[:,0,final_flavor,0], color=std_osc_color, linewidth=4, label=label, linestyle=linestyles[rho] )


    # Decoherence
    calculator.set_decoherence_D_matrix(D_matrix_eV=D_matrix_eV, n=n, E0_eV=E0_eV)
    for i, initial_rho in enumerate(rho_values) :
        osc_probs = calculator.calc_osc_prob(
            initial_flavor=initial_flavor,
            initial_rho=initial_rho,
            energy_GeV=E_GeV,
            distance_km=L_km,
        )
        label = calculator.get_transition_prob_tex(initial_flavor=initial_flavor, final_flavor=final_flavor, initial_flavor_nubar=(rho==1), final_flavor_nubar=(rho==1))
        ax.plot( E_GeV, osc_probs[:,0,final_flavor,0], color=decoh_color, linewidth=4, label=label, linestyle=linestyles[i] )

    # Format
    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_xlim(E_GeV[0],E_GeV[-1])
    ax.set_ylim(0.,1.)
    fig.tight_layout()


    #
    # DUNE case
    #

    #TODO


def plot_arxiv_1811_04982(num_points=1000) :
    '''
    Reproduce plots from arXiv 1811.04982 (DUNE)

    Note that the authors claim the matter effects are insignificant for the mu->mu channel,
    but I do not find this to be true in the presence of CPT violating decoherence operators.
    '''

    print("\n>>> Plot arXiv 1811.04982")


    #
    # Create calculator
    #

    init_kw = {}
    if TOOL == "nusquids" :
        init_kw["energy_nodes_GeV"] = np.logspace(-2., 2., num=1000)

    calculator = OscCalculator(
        tool=TOOL,
        atmospheric=False,
        num_neutrinos=3,
        **init_kw
    )

    # Define system (Tabe III)
    L_km = 1300.
    calculator.set_mixing_angles( np.deg2rad(33.63), np.deg2rad(8.52), np.deg2rad(48.7), deltacp=(3.*np.pi/2.)) # CP-phase for fig 1
    calculator.set_mass_splittings( 7.4e-5, 2.515e-3 )
    # calculator.set_matter(matter="constant", matter_density_g_per_cm3=2.75, electron_fraction=0.5) #TODO Get a DUNE number
    calculator.set_matter("vacuum") #TODO

    # Decoherence config
    calculator.set_calc_basis("nxn")


    #
    # Figure 1
    #

    # Energy range
    E_GeV = np.geomspace( 0.1, 20., num=num_points)

    # Define decoherence matrix
    n = 0
    E0_eV = 1e9

    gamma_eV = 1e-23 * 1e9
    beta28_eV = gamma_eV / np.sqrt(3.)
    beta12_eV = gamma_eV / 3.
    beta56_eV = gamma_eV / np.sqrt(3.)
    beta47_eV = gamma_eV / np.sqrt(3.)

    diag_matrix = np.diag([0] + [gamma_eV]*8) # State selection case

    cases = collections.OrderedDict()

    # D = diag_matrix.copy()
    # D_matrices_eV["Diagonal"] = D

    D = diag_matrix.copy()
    D[1,2] = beta12_eV
    D[2,1] = beta12_eV
    cases[r"$\beta_{12}$"] = {
        "D_matrix_eV" : D,
        "color" : "black",
        "linestyle" : "-",
    }

    D = diag_matrix.copy()
    D[2,8] = beta28_eV
    D[8,2] = beta28_eV
    cases[r"$\beta_{28}$"] = {
        "D_matrix_eV" : D,
        "color" : "red",
        "linestyle" : "-",
    }

    D = diag_matrix.copy()
    D[4,7] = beta47_eV
    D[7,4] = beta47_eV
    cases[r"$\beta_{47}$"] = {
        "D_matrix_eV" : D,
        "color" : "purple",
        "linestyle" : "--",
    }

    D = diag_matrix.copy()
    D[5,6] = beta56_eV
    D[6,5] = beta56_eV
    cases[r"$\beta_{56}$"] = {
        "D_matrix_eV" : D,
        "color" : "blue",
        "linestyle" : "--",
    }

    # Choose channels
    initial_flavor = 1
    initial_rho_values = [0, 1] # nu, nubar
    final_flavor_values = [ 0, 1 ] # e, mu

    # Create the fig
    fig, ax = plt.subplots( nrows=3, ncols=len(final_flavor_values), figsize=(10, 12) )

    print("+++", np.shape(ax))

    # Loop over channels
    for i_flav, final_flavor in enumerate(final_flavor_values) :

        # Std oscillations
        # Note that this is not in the figure in the paper
        calculator.set_std_osc()
        delta_osc_probs_cpt = None
        for rho in initial_rho_values :
            osc_probs = calculator.calc_osc_prob(
                initial_flavor=initial_flavor,
                initial_rho=rho,
                energy_GeV=E_GeV,
                distance_km=L_km,
            )[:,0,final_flavor]
            ax[rho, i_flav].plot( E_GeV, osc_probs, color="grey", linewidth=3, label="Std osc", linestyle=":" )
            delta_osc_probs_cpt = osc_probs if delta_osc_probs_cpt is None else (delta_osc_probs_cpt - osc_probs)
        ax[2, i_flav].plot( E_GeV, delta_osc_probs_cpt, color="grey", linewidth=3, label="Std osc", linestyle=":" )

        # Decoherence
        for i, (label, case_dict) in enumerate(cases.items()) :
            calculator.set_decoherence_D_matrix(D_matrix_eV=case_dict["D_matrix_eV"], n=n, E0_eV=E0_eV)
            delta_osc_probs_cpt = None
            for rho in initial_rho_values :
                osc_probs = calculator.calc_osc_prob( #TODO A-CPT function
                    initial_flavor=initial_flavor,
                    initial_rho=rho,
                    energy_GeV=E_GeV,
                    distance_km=L_km,
                )[:,0,final_flavor]
                ax[rho, i_flav].plot( E_GeV, osc_probs, color=case_dict["color"], linewidth=3, label=label, linestyle=case_dict["linestyle"] )
                delta_osc_probs_cpt = osc_probs if delta_osc_probs_cpt is None else (delta_osc_probs_cpt - osc_probs)
            ax[2, i_flav].plot( E_GeV, delta_osc_probs_cpt, color=case_dict["color"], linewidth=3, label=label, linestyle=case_dict["linestyle"] )

            print(f"+++ decoh case {i} done ({label})")

        # Format
        for rho in initial_rho_values :
            for final_flavor in final_flavor_values :
                ax[rho, final_flavor].set_ylabel( r"$%s$"%calculator.get_transition_prob_tex(initial_flavor,final_flavor,rho,rho) )
            ax[rho, 0].set_ylim(0., 0.3)
            ax[rho, 1].set_ylim(0., 1.)
        for final_flavor in final_flavor_values :
            ax[final_flavor, 2].set_ylabel( r"$\Delta P_{CPTV}$" )
        ax[0, 2].set_ylim(-0.1, +0.1)
        ax[1, 2].set_ylim(-0.1, +0.1)
        for this_ax in ax :
            this_ax.set_xlabel(r"$E$ [GeV]")
            this_ax.set_xscale("log")
            this_ax.set_xlim(E_GeV[0],E_GeV[-1])
            this_ax.legend(fontsize=8)
        fig.tight_layout()



def plot_atmospheric_1d(
    num_points=1000,
) :
    '''
    Playing with 1D atmospheric neutrino scenarios scenarios
    '''

    print("\n>>> Plot atmospheric 1D oscillations")

    #
    # Create calculator
    #

    # Choose baseline, energy, etc
    L_km = EARTH_DIAMETER_km
    # E_GeV = np.linspace(6., 120., num=num_points)
    E_GeV = np.logspace(0., 5., num=num_points)

    # Create calc
    init_kw = {}
    if TOOL == "nusquids" :
        init_kw["energy_nodes_GeV"] = E_GeV
    calculator = OscCalculator(
        tool=TOOL,
        atmospheric=False,
        num_neutrinos=3,
        **init_kw
    )

    calculator.set_calc_basis("nxn")
    calculator.set_matter("vacuum") #TODO Use an Earth model (take a slice of a oscillogram)


    #
    # Decoherence matrix cases
    #

    # Using the cases from arXiv 1811.04982 Fig. 1

    n = 0

    E0_eV = 1e9

    # gamma_eV = 1e-23 * 1e9
    gamma_eV = REF_GAMMA_eV

    beta28_eV = gamma_eV / np.sqrt(3.)
    beta12_eV = gamma_eV / 3.
    beta56_eV = gamma_eV / np.sqrt(3.)
    beta47_eV = gamma_eV / np.sqrt(3.)

    diag_matrix = np.diag([0] + [gamma_eV]*8) # State selection case

    D_matrices_eV = collections.OrderedDict()

    D = diag_matrix.copy()
    D[1,2] = beta12_eV
    D[2,1] = beta12_eV
    D_matrices_eV[r"$\beta_{12}$"] = D

    D = diag_matrix.copy()
    D[2,8] = beta28_eV
    D[8,2] = beta28_eV
    D_matrices_eV[r"$\beta_{28}$"] = D

    D = diag_matrix.copy()
    D[4,7] = beta47_eV
    D[7,4] = beta47_eV
    D_matrices_eV[r"$\beta_{47}$"] = D

    D = diag_matrix.copy()
    D[5,6] = beta56_eV
    D[6,5] = beta56_eV
    D_matrices_eV[r"$\beta_{56}$"] = D

    D = diag_matrix.copy()
    D[1,2] = beta12_eV
    D[2,1] = beta12_eV
    D[2,8] = beta28_eV
    D[8,2] = beta28_eV
    D[4,7] = beta47_eV
    D[7,4] = beta47_eV
    D[5,6] = beta56_eV
    D[6,5] = beta56_eV
    D_matrices_eV[r"$\beta_{12,28,47,56}$"] = D

    # D_matrices_eV["Optimal"] = np.diag([gamma_eV]*9) # 1811.04982 Section C.5
    # D_matrices_eV["Optimal"][2,8] = gamma_eV / np.sqrt(3)
    # D_matrices_eV["Optimal"][8,2] = D_matrices_eV["Optimal"][2,8]
    # D_matrices_eV["Optimal"][1,2] = np.sqrt(2./3.) * gamma_eV
    # D_matrices_eV["Optimal"][2,1] = D_matrices_eV["Optimal"][1,3]
    # D_matrices_eV["Optimal"][5,6] = gamma_eV / 3.
    # D_matrices_eV["Optimal"][6,5] = D_matrices_eV["Optimal"][5,6]
    # D_matrices_eV["Optimal"][4,7] = -gamma_eV / 3.
    # D_matrices_eV["Optimal"][7,4] = D_matrices_eV["Optimal"][4,7]


    #
    # Plot
    #

    # Choose baseline, energy, etc
    initial_flavor = 1
    final_flavors = [ 0, 1 ] 
    initial_rho_values = [ 0, 1 ]

    # Create the fig
    fig = Figure( ny=3, nx=2, figsize=(10, 12) )

    # Plot steering
    linestyles = ["-", "--", "-.", ":", ":"]

    # Loop over channels
    for final_flavor in final_flavors :

        # Std oscillations
        calculator.set_std_osc()
        delta_osc_probs_cpt = None
        for rho in initial_rho_values :
            osc_probs = calculator.calc_osc_prob(
                initial_flavor=initial_flavor,
                initial_rho=rho,
                energy_GeV=E_GeV,
                distance_km=L_km,
            )[:,0,final_flavor]
            fig.get_ax(x=final_flavor, y=rho).plot( E_GeV, osc_probs, color="grey", linewidth=3, label="Std osc", linestyle="-" )
            delta_osc_probs_cpt = osc_probs if delta_osc_probs_cpt is None else (delta_osc_probs_cpt - osc_probs)
        fig.get_ax(x=final_flavor, y=2).plot( E_GeV, delta_osc_probs_cpt, color="grey", linewidth=3, label="Std osc", linestyle="-" )

        # Decoherence
        for i, (label, D_matrix_eV) in enumerate(D_matrices_eV.items()) :
            print(D_matrix_eV)
            calculator.set_decoherence_D_matrix(D_matrix_eV=D_matrix_eV, n=n, E0_eV=E0_eV)
            delta_osc_probs_cpt = None
            for rho in initial_rho_values :
                osc_probs = calculator.calc_osc_prob(
                    initial_flavor=initial_flavor,
                    initial_rho=rho,
                    energy_GeV=E_GeV,
                    distance_km=L_km,
                )[:,0,final_flavor]
                fig.get_ax(x=final_flavor, y=rho).plot( E_GeV, osc_probs, color=colors[i], linewidth=3, label=label, linestyle=linestyles[i] )
                delta_osc_probs_cpt = osc_probs if delta_osc_probs_cpt is None else (delta_osc_probs_cpt - osc_probs)
            fig.get_ax(x=final_flavor, y=2).plot( E_GeV, delta_osc_probs_cpt, color=colors[i], linewidth=3, label=label, linestyle=linestyles[i] )

        # Format
        for rho in initial_rho_values :
            for final_flavor in final_flavors :
                fig.get_ax(x=final_flavor, y=rho).set_ylabel( r"$%s$"%calculator.get_transition_prob_tex(initial_flavor,final_flavor,rho,rho) )
            fig.get_ax(x=0, y=rho).set_ylim(0., 0.3)
            fig.get_ax(x=1, y=rho).set_ylim(0., 1.)
        for final_flavor in final_flavors :
            fig.get_ax(x=final_flavor, y=2).set_ylabel( r"$\Delta P_{CPTV}$" )
        fig.get_ax(x=0, y=2).set_ylim(-0.1, +0.1)
        fig.get_ax(x=1, y=2).set_ylim(-0.05, +0.05)
        fig.quick_format( xlabel=r"$E$ [GeV]", xscale="log", legend_kw={"fontsize":8}, xlim=(E_GeV[0],E_GeV[-1]) )


#
# Main
#

if __name__ == "__main__" :

    from utils.script_tools import ScriptWrapper
    from utils.filesys_tools import replace_file_ext
    with ScriptWrapper( replace_file_ext(__file__,".log") ) as script :

        num_points = 10

        # plot_arxiv_1807_07823(num_points=num_points)

        plot_arxiv_1811_04982(num_points=num_points)

        # plot_atmospheric_1d(num_points=num_points)

        print("")
        dump_figures_to_pdf( replace_file_ext(__file__,".pdf") )
