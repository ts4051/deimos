'''
Comparing a range of QG-motivated decoherence scenarios

This is part of addressing comments on the review of the MEOWS decoherence paper

Refs:
  [1] https://arxiv.org/pdf/2306.14778.pdf
  [2] https://arxiv.org/abs/2208.12062

Tom Stuttard
'''

import sys, os, collections, copy, datetime

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *

from deimos.density_matrix_osc_solver.density_matrix_osc_solver import km_to_eV, GeV_to_eV
from deimos.utils.oscillations import calc_disappearance_prob_2flav_vacuum


#
# Globals
#

SOLVER = "deimos"

NUM_SCAN_POINTS = 100


#
# Helper functions
#

def plot_LoE(ax, E_GeV, L_km, P, color, linestyle, label=None) :
    '''
    Plot an L/E distribution
    '''

    # Calc L/E
    LoE = L_km / E_GeV

    # Check P
    #TODO handle 2D
    assert P.ndim == 1

    # Plot
    ax.plot(LoE, P, color=color, linestyle=linestyle, label=label)

    # Format ax
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(LoE[-1], LoE[0]) # Reverse x axis (since L/E is inverse of E)
    ax.set_xlabel("%s / %s" % (DISTANCE_LABEL, ENERGY_LABEL))

   
def calc_qg_damping_coefficient(qg_model, E_GeV, mass_splitting_eV2, theta_rad, **qg_params) :
    '''
    Calculate the damping coefficient for the various models considered, alpha

    Definition : D = alpha * L
    Damping term: exp{-D}

    This is only valid for the 2-flavor vacuum case
    '''

    # Convert units
    E_eV = E_GeV * GeV_to_eV

    # Get QG scale
    E_qg_eV = qg_params["E_qg_eV"]

    # Checks
    assert E_qg_eV is not None
    assert qg_model is not None
    assert isinstance(qg_model, str)

    # Some elements are common to multiple models
    if qg_model in ["minimal_length_fluctuations", "metric_fluctuations"] :

        # Mass of lightest neutrino state is a free parameter
        m1_eV = qg_params["m1_eV"]
        m2_eV = np.sqrt(  mass_splitting_eV2 + np.square(m1_eV) ) #TODO think I might have made a function for this already? if so use it, it not then make one

        # Calc average group velocity, p/E, where p is average of the two states
        # Using E^2 = p^2 + m^2 to get p, where again m is the average
        # Note that this is basically always = 1 for the relevent energies (as also mentioned in footnote of page 12 of https://arxiv.org/pdf/2306.14778.pdf)
        #TODO cross-check calc via Lorentz boost method?
        m_mean_eV = ( m1_eV + m2_eV ) / 2.
        p_mean_eV = np.sqrt( np.square(E_eV) - np.square(m_mean_eV) )
        v_g = p_mean_eV / E_eV
        
        # Calc (delta m)^2   --> NOT delta (m^2)   (e.g. NOT mass splitting)
        dm_squared_eV = np.square( m2_eV - m1_eV )  #TODO try alt calc
    
    # Calc damping term: Minimal length fluctuations case
    if qg_model == "minimal_length_fluctuations" :
        alpha = 16. * np.power(E_eV, 4.) * dm_squared_eV
        alpha /= (v_g * np.power(E_qg_eV, 5.) )

    # Calc damping term: Stochastic fluctuations of the metric case
    #TODO eqn 29 vs 30? think 30 is just a simplification of 29 in some cases. For now using eqn 29
    elif qg_model == "metric_fluctuations" :
        alpha = 1. / ( 8. * v_g * E_qg_eV )
        alpha *= np.square( 1. + ( np.square(E_eV) / np.square(m1_eV*m2_eV) ) ) 
        alpha *= dm_squared_eV

    # Calc damping term: nu-VBH interactions
    elif qg_model == "nu_vbh_interactions" : #TODO which interaction type?
        # https://arxiv.org/pdf/2007.00068.pdf eqn 20, with zeta = 1 for natural scale (and using E_qg as free param as with other models here, e.g. not necessarily equal to M_P)
        n = qg_params["n"]
        alpha = np.power(E_eV, n) / np.power(E_qg_eV, n-1.)

    else :
        raise Exception("Unknown model : %s" % qg_model)

    return alpha


def calc_coherence_length(*args, **kwargs) :

    '''
    Get the coherence length for a given QG model

    This is when exp{-alpah*L} = 1/e   ->  -alpha*L = -1   -> L_coh = 1/alpha
    '''
    return 1. / calc_qg_damping_coefficient(*args, **kwargs)


def calc_2flav_survival_prob(
    calculator, 
    E_GeV, 
    L_km, 
    flavor, 
    nubar=False, 
    # Solver options
    use_deimos=False,
    # QG model
    qg_model=None, 
    **qg_params
) :
    '''
    Equation 24 from [1], which gives 2-flavor vacuum surival probablity with a damping factor included
    '''

    #TODO Just implement D[rho] operator instead...

 
    # This is for 2 flavor only
    assert calculator.num_neutrinos == 2

    # This is for vacuum only
    #TODO


    # Extract standard osc params
    mass_splitting_eV2 = calculator.get_mass_splittings()
    assert len(mass_splitting_eV2) == 1
    mass_splitting_eV2 = mass_splitting_eV2[0]

    theta_rad = calculator.get_mixing_angles()
    assert len(theta_rad) == 1
    theta_rad = theta_rad[0]


    #
    # Calc standard osc
    #

    if use_deimos : #TODO not sure I am getting the correct answer here when using my own solver, why?

        calculator.set_std_osc()

        Pstd = calculator.calc_osc_prob(
            energy_GeV=E_GeV,
            initial_flavor=flavor,
            distance_km=L_km,
            nubar=nubar,
        )

        # Remove distance dimension   #TODO make more general
        Pstd = Pstd[:,0,...]

        # Get final flavor
        Pstd = Pstd[...,flavor]

    else :

        Pstd = 1. - calc_disappearance_prob_2flav_vacuum(E_GeV=E_GeV, L_km=L_km, mass_splitting_eV2=mass_splitting_eV2, theta_rad=theta_rad)


    # Done now if user doesn't ant to add QG effects
    if qg_model is None :
        return Pstd


    #TODO could also use simple analytic expression


    #
    # Calc damping term
    #

    # Get the coefficient
    alpha = calc_qg_damping_coefficient(E_GeV=E_GeV, mass_splitting_eV2=mass_splitting_eV2, theta_rad=theta_rad, qg_model=qg_model, **qg_params)

    # Convert units
    L_eV = L_km * km_to_eV

    # Calc damping term
    D = alpha * L_eV
    damping_term = np.exp(-D)


    #
    # Calc overall survival prob.
    #

    P_qg = ( damping_term * Pstd ) + ( (1. - damping_term) * (1. - (np.square(np.sin(2.*theta_rad)) / 2.)) )

    return P_qg

    
def generate_random_D_matrix(calculator, random_state, disable_relaxation_params=False, disable_decoh_params=False, enforce_12_45_67_pairs=False) :
    '''
    Generate a random D matrix, and check if it is valid
    '''

    #TODO support beta terms

    #TODO doesn't really need to use calculator, need to move check_decoherence_D_matrix outside of the class

    # Generate a random diagonal D matrix
    D_diag = random_state.uniform(0., 1, size=9)
    D_diag[0] = 0.

    # Disable relaxation params, if requested
    if disable_relaxation_params :
        D_diag[3] = 0.
        D_diag[8] = 0.

    # Disable decoherence params, if requested
    if disable_decoh_params :
        D_diag[1] = 0.
        D_diag[2] = 0.
        D_diag[4] = 0.
        D_diag[5] = 0.
        D_diag[6] = 0.
        D_diag[7] = 0.

    # Enforce the 12/45/67 pair equalities, if requested
    if enforce_12_45_67_pairs :
        D_diag[2] = D_diag[1]
        D_diag[5] = D_diag[4]
        D_diag[7] = D_diag[6]

    # Form the full matrix
    D = np.diag(D_diag.tolist())

    # Test which cases are valid
    try :
        calculator.check_decoherence_D_matrix(D)
        valid = True
    except Exception as e:
        # print(str(e))
        valid = False

    return D, valid


#
# Plot functions
#

def reproduce_2306_14778() :
    '''
    Reproducing plots from https://arxiv.org/pdf/2306.14778.pdf to test my implementation of their scenarioa
    '''


    #
    # Create model
    #

    # Create calculator
    # calculator = OscCalculator(
    #     tool=SOLVER,
    #     atmospheric=False,
    #     num_neutrinos=3,
    # )

    # Create calculator
    # Using 2-flavor approx
    calculator = OscCalculator(
        tool=SOLVER,
        atmospheric=False,
        num_neutrinos=2,
    )

    # Use vacuum
    calculator.set_matter("vacuum")




    #
    # Fig 1 : KamLAND
    #

    # This is for KamLAND
    L_km = KAMLAND_BASELINE_km
    LoE_values_km_per_MeV = np.linspace(20., 105, num=NUM_SCAN_POINTS) # Match range of Figure
    E_GeV_values = np.sort( L_km / (LoE_values_km_per_MeV*1e3) )
    initial_flavor, final_flavor, nubar = 0, 0, True # antinu survival

    # Set osc params from figure caption
    calculator.set_mixing_angles( np.arcsin(np.sqrt(0.85)) / 2. )
    calculator.set_mass_splittings( 7.53e-5 )

    # Calculations assume survival
    assert initial_flavor == final_flavor

    # Make fig
    fig1, ax1 = plt.subplots( figsize=(6, 4) )

    # Plot std osc
    Pstd = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=None)
    plot_LoE(ax=ax1, E_GeV=E_GeV_values, L_km=L_km, P=Pstd, color="red", linestyle="--", label="Std osc")

    #TODO REMOVE
    # Pstd = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=None, use_deimos=True)
    # plot_LoE(ax=ax1, E_GeV=E_GeV_values, L_km=L_km, P=Pstd, color="green", linestyle="-.", label="Std osc (DEIMOS)")

    # Plot QG
    m1_eV = 1. # From caption
    E_qg_eV = 1e24 * GeV_to_eV # From caption
    qg_model = "metric_fluctuations"




    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    E_qg_eV *= 1e-5
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....
    #TODO I need a correction to reproduce plots, why? Seems fine for the SUperK case without a correction....






    Pqg = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=E_qg_eV, m1_eV=m1_eV, qg_model=qg_model)
    plot_LoE(ax=ax1, E_GeV=E_GeV_values, L_km=L_km, P=Pqg, color="blue", linestyle=":", label="QG")

    # Format
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()



    #
    # Fig 3 : SuperK
    #

    # This is for KamLAND
    L_km = 10. # From caption
    LoE_values_km_per_GeV = np.geomspace(1., 1e4, num=NUM_SCAN_POINTS) # Match range of Figure
    E_GeV_values = np.sort( L_km / (LoE_values_km_per_GeV) )
    initial_flavor, final_flavor, nubar = 0, 0, True # antinumu survival

    # Set osc params from figure caption
    calculator.set_mixing_angles( np.arcsin(np.sqrt(0.99)) / 2. )
    calculator.set_mass_splittings( 2.45e-3 )

    # Calculations assume survival
    assert initial_flavor == final_flavor

    # Make fig
    fig3, ax3 = plt.subplots( figsize=(6, 4) )

    # Plot std osc
    Pstd = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=None)
    plot_LoE(ax=ax3, E_GeV=E_GeV_values, L_km=L_km, P=Pstd, color="red", linestyle="--", label="Std osc")


    # #TODO REMOVE
    # Pstd = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=None, use_deimos=True)
    # plot_LoE(ax=ax3, E_GeV=E_GeV_values, L_km=L_km, P=Pstd, color="green", linestyle="-.", label="Std osc (DEIMOS)")


    # Plot QG case from paper
    E_qg_eV = 1e30 * GeV_to_eV # From caption
    qg_model = "metric_fluctuations"
    Pqg = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=E_qg_eV, m1_eV=m1_eV, qg_model=qg_model)
    plot_LoE(ax=ax3, E_GeV=E_GeV_values, L_km=L_km, P=Pqg, color="blue", linestyle=":", label=qg_model.replace("_", " ").title())

    # Add another QG case of interest
    E_qg_eV = 1e0 * GeV_to_eV # Choosing something with reasonable signal
    qg_model = "minimal_length_fluctuations"
    Pqg = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=E_qg_eV, m1_eV=m1_eV, qg_model=qg_model)
    plot_LoE(ax=ax3, E_GeV=E_GeV_values, L_km=L_km, P=Pqg, color="magenta", linestyle=":", label=qg_model.replace("_", " ").title())

    # Format
    ax3.set_xscale("log")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()

    # Cross check standard oscillogram
    if False :
        E_GeV = np.geomspace(0.1, 100., num=100)
        coszen = np.linspace(-1., 1., num=100)
        E_GeV_grid, coszen_grid = np.meshgrid(E_GeV, coszen, indexing="ij")
        L_km_grid = calc_path_length_from_coszen(coszen)
        P = calc_disappearance_prob_2flav_vacuum(E_GeV=E_GeV_grid, L_km=L_km_grid, mass_splitting_eV2=calculator.get_mass_splittings()[0], theta_rad=calculator.get_mixing_angles()[0])
        fig, ax = plt.subplots(figsize=(7,7))
        plot_colormap( ax=ax, x=E_GeV, y=coszen, z=P, vmin=0., vmax=1., cmap="jet" )
        ax.set_xscale("log")
        fig.tight_layout()


def compare_qg_models() :
    '''
    Compare decoherence resulting from a range on QG models, for the case of atmospheric neutrinos
    '''

    pass


def compare_qg_models_coherence_length() :
    '''
    Compare the natural coherence length for a range of QG models
    '''

    # Steer standard osc physics
    mass_splitting_eV2 = MASS_SPLITTINGS_eV2[-1] # atmo
    mixing_angle_rad = MIXING_ANGLES_rad[-1] # atmo

    # Steer QG
    E_qg_eV = PLANCK_MASS_eV
    m1_eV = 1.

    # Decide E range
    E_eV = np.logspace(0., 30, num=100)

    # Make figure
    fig, ax = plt.subplots( figsize=(6, 4) )

    # Mark Planck scale
    ax.axvline(PLANCK_MASS_eV, linestyle="-", lw=1, color="black", label="Planck scale")
    ax.axhline(PLANCK_LENGTH_m, linestyle="-", lw=1, color="black")

    # Mark Earth diameter
    ax.axhline(EARTH_DIAMETER_km*1e3, linestyle="-", lw=1, color="brown", label="Earth diameter")

    # Calc and plot coherence length for each model
    L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="minimal_length_fluctuations", E_qg_eV=E_qg_eV, m1_eV=m1_eV) * 1e3 # km -> m
    ax.plot(E_eV, L_coh_m, color="red", label="Minimal length fluctuations", linestyle="-", lw=2)

    L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="metric_fluctuations", E_qg_eV=E_qg_eV, m1_eV=m1_eV) * 1e3 # km -> m
    ax.plot(E_eV, L_coh_m, color="blue", label="Metric fluctuations", linestyle="-", lw=2)

    for n, linestyle in zip([0, 1, 2, 3], ["-", "--", "-.", ":"]) :
        L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="nu_vbh_interactions", E_qg_eV=E_qg_eV, n=n) * 1e3 # km -> m
        ax.plot(E_eV, L_coh_m, color="orange", label=r"$\nu$-VBH ($n$=%i)"%n, linestyle=linestyle, lw=2)

    # Format
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_{\nu}$ [eV]")
    ax.set_ylabel(r"$L_{coh}$ [m]")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()



def compare_D_matrix_textures() :
    '''
    Compare decoherence resulting from different textures of the D matrix

    (a) Diagonal
    (b) Off-diagonal

    Respecting all constraints between parameters
    '''


    #
    # Create model
    # 

    calculator = OscCalculator(
        tool=SOLVER,
        atmospheric=True,
        num_neutrinos=3,
    )

    # Vacuum
    calculator.set_matter("vacuum")

    # Define plot range
    initial_flavor, final_flavor = 1, 1
    nubar = False
    E_GeV_values = np.logspace(2., 5., num=NUM_SCAN_POINTS)
    coszen_values = np.linspace(-1, 0., num=NUM_SCAN_POINTS)
    coszen_ref = -1.


    #
    # Loop over n
    #

    E0_eV = 1e12 # 1 TeV

    n_cases = [
        {
            "n" : 0,
            "E_GeV_ref" : 1e3,
            "gamma0_eV" : 1.18e-15 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
        {
            "n" : 2,
            "E_GeV_ref" : 1e5,
            "gamma0_eV" : 9.80e-18 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
    ]

    for n_case in n_cases :

        print("n = %i" % n_case["n"])


        #
        # Non-relaxation term (phase perturbation cases)
        #

        # Define cases
        gamma_cases = [
            {
                "D_matrix_eV" : None,
                "color" : "grey",
                "linestyle" : "-",
                "label" : "Standard osc",
            },
            {
                "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
                # "gamma_scaling" : 1,
                "color" : "black",
                "linestyle" : "-",
                # "label" : r"$\Gamma_1 = \Gamma_2 = \Gamma_4 = \Gamma_5 = \Gamma_6 = \Gamma_7 = \Gamma$",
                "label" : r"Phase perturbation (PP)",
            },
            {
                "D_matrix_eV" : np.diag([0., 0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
                # "gamma_scaling" : 1.1,
                "color" : "magenta",
                "linestyle" : "--",
                "label" : r"PP w/ $\Gamma_1 = \Gamma_2 = 0$",
            },
            {
                "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
                # "gamma_scaling" : 1.1,
                "color" : "orange",
                "linestyle" : "--",
                "label" : r"PP w/ $\Gamma_4 = \Gamma_5 = 0$",
            },
            {
                "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0.]),
                # "gamma_scaling" : 4.,
                "color" : "dodgerblue",
                "linestyle" : "--",
                "label" : r"PP w/ $\Gamma_6 = \Gamma_7 = 0$",
            },
            # { # Forbidden
            #     "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0., 0., 0., 0.]),
            #     "color" : "magenta",
            #     "linestyle" : "--",
            #     "label" : r"$\Gamma_1 = \Gamma_2 = \Gamma$, rest = 0",
            # },
            # {
            #     "D_matrix_eV" : np.diag([0.,0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0.]),
            #     "color" : "magenta",
            #     "linestyle" : "--",
            #     "label" : r"$\Gamma_4 = \Gamma_5 = \Gamma$, rest = 0",
            # },
            # {
            #     "D_matrix_eV" : np.diag([0., 0., 0., 0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
            #     "color" : "magenta",
            #     "linestyle" : "--",
            #     "label" : r"$\Gamma_6 = \Gamma_7 = \Gamma$, rest = 0",
            # },
        ]

        # Make figure
        fig_E, ax_E = None, None
        fig_cz, ax_cz = None, None

        # Loop over cases
        for gamma_case in gamma_cases :

            print(gamma_case["label"])

            # Set model
            if gamma_case["D_matrix_eV"] is None :
                calculator.set_std_osc()
            else :
                calculator.set_decoherence_D_matrix(D_matrix_eV=gamma_case["D_matrix_eV"], n=n_case["n"], E0_eV=E0_eV)

            # Plot vs coszen
            fig_cz, ax_cz, _ = calculator.plot_osc_prob_vs_distance( 
                initial_flavor=initial_flavor,
                final_flavor=final_flavor,
                energy_GeV=n_case["E_GeV_ref"], 
                coszen=coszen_values, 
                nubar=nubar, 
                fig=fig_cz, 
                ax=ax_cz, 
                label=gamma_case["label"], 
                color=gamma_case["color"], 
                linestyle=gamma_case["linestyle"], 
                xscale="linear",
                ylim=(-0.05, 1.05),
                lw=2,
            )

            # Plot vs E
            fig_E, ax_E, _ = calculator.plot_osc_prob_vs_energy( 
                initial_flavor=initial_flavor,
                final_flavor=final_flavor,
                energy_GeV=E_GeV_values, 
                coszen=coszen_ref, 
                nubar=nubar, 
                fig=fig_E, 
                ax=ax_E, 
                label=gamma_case["label"], 
                color=gamma_case["color"], 
                linestyle=gamma_case["linestyle"], 
                xscale="log",
                ylim=(-0.05, 1.05),
                lw=2,
            )

        # Titles
        fig_E.suptitle(r"$n$ = %i, $\Gamma_0$ = %0.3g eV, $\cos(\theta)$ = %0.3g" % (n_case["n"], n_case["gamma0_eV"], coszen_ref) )
        fig_cz.suptitle(r"$n$ = %i, $\Gamma_0$ = %0.3g eV, $E_\nu$ = %0.3g GeV" % (n_case["n"], n_case["gamma0_eV"], n_case["E_GeV_ref"]) )

        # Format
        for ax in [ax_E, ax_cz] :
            ax[0].legend(fontsize=10)
        for fig in [fig_E, fig_cz] :
            fig.tight_layout()



        #
        # Including relaxation terms (state selection cases)
        #

        #TODO


def compare_random_D_matrix_textures(num_models=100) :
    '''
    Generate random (valid) D matrix textures and compare them
    '''

    start_time = datetime.datetime.now()

    # Init RNG
    random_state = np.random.RandomState(12345)


    #
    # Create model
    # 

    calculator = OscCalculator(
        tool=SOLVER,
        atmospheric=True,
        num_neutrinos=3,
    )

    # Vacuum
    calculator.set_matter("vacuum")

    # Define plot range
    initial_flavor, final_flavor = 1, 1
    nubar = False
    E_GeV_values = np.geomspace(500, 10e3, num=NUM_SCAN_POINTS)
    coszen_values = np.linspace(-1, 0., num=NUM_SCAN_POINTS)
    coszen_ref = -1.


    #
    # Loop over n
    #

    E0_eV = 1e12 # 1 TeV

    n_cases = [
        {
            "n" : 0,
            "E_GeV_ref" : 1e3,
            "gamma0_eV" : 1.18e-15 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
        {
            "n" : 2,
            "E_GeV_ref" : 1e5,
            "gamma0_eV" : 9.80e-18 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
    ]

    for n_case in n_cases :

        print("n = %i" % n_case["n"])


        #
        # Plot our cases
        #

        # Make figures
        fig, ax = plt.subplots(ncols=2, figsize=(10,4))

        # Plot helper function
        def _plot(color, linestyle, lw, zorder, label=None, alpha=1.) :
            # Common args
            plot_kw = dict(
                initial_flavor=initial_flavor,
                final_flavor=final_flavor,
                nubar=nubar, 
                ylim=(-0.05, 1.05),
                label=label,
                color=color,
                linestyle=linestyle,
                lw=lw,
                zorder=zorder,
                alpha=alpha,
            )
            # Plot vs coszen
            calculator.plot_osc_prob_vs_distance( 
                energy_GeV=n_case["E_GeV_ref"], 
                coszen=coszen_values, 
                fig=fig, 
                ax=[ax[0]], 
                xscale="linear",
                **plot_kw,
            )
            # Plot vs E
            calculator.plot_osc_prob_vs_energy( 
                energy_GeV=E_GeV_values, 
                coszen=coszen_ref, 
                fig=fig, 
                ax=[ax[1]], 
                xscale="log",
                **plot_kw,
            )

        # Plot standard osc
        calculator.set_std_osc()
        _plot(color="grey", linestyle="-", lw=3, label="Standard osc", zorder=5)

        # # Plot phase perturbation model
        calculator.set_decoherence_model("randomize_phase", gamma0_eV=n_case["gamma0_eV"], n=n_case["n"], E0_eV=E0_eV)
        phase_perturbation_color = "red"
        _plot(color=phase_perturbation_color, linestyle="-", lw=3, label="Phase perturbation", zorder=5)

        # # Plot state selection
        calculator.set_decoherence_model("randomize_state", gamma0_eV=n_case["gamma0_eV"], n=n_case["n"], E0_eV=E0_eV)
        state_selection_color = "blue"
        _plot(color=state_selection_color, linestyle="-", lw=3, label="State selection", zorder=5)


        #
        # Generate random models and plot resulting transition probabilities
        #

        #TODO Better to analytically find bounding cases if possible, but difficult as system gains DOFs

        # Handle both with and without relaxation params cases           #TODO beta
        # for (disable_relaxation_params, enforce_12_45_67_pairs, color) in [ (True, True, phase_perturbation_color), (False, False, state_selection_color) ] :
        for (disable_relaxation_params, enforce_12_45_67_pairs, color) in [ (False, False, state_selection_color), (True, True, phase_perturbation_color) ] :

            # Choose whether to plot individual models, or the envelope
            plot_envelope = True

            # Loop to generate models
            trial_counter, model_counter = 0, 0
            ymin_E, ymax_E, ymin_cz, ymax_cz = None, None, None, None
            while model_counter < num_models :

                # Report
                print("Trial %i (valid models so far = %i of %i)" % (trial_counter, model_counter, num_models))
                trial_counter += 1

                # Generate random D matrix
                D, valid = generate_random_D_matrix(
                    calculator=calculator,
                    random_state=random_state, 
                    disable_relaxation_params=disable_relaxation_params, 
                    disable_decoh_params=False, 
                    enforce_12_45_67_pairs=enforce_12_45_67_pairs,
                )

                # Only proceed if valid
                if valid :

                    # Book-keeping
                    # model_color = cmap(float(model_counter)/float(num_models))
                    # model_color = "seagreen"
                    model_counter += 1

                    # Scale such that the largest D matrix element = gamma0
                    D = D * n_case["gamma0_eV"] * (1. / np.max(D))    #TODO how exactly to do this? Is this coherence length of largest element? think so

                    # Set model
                    calculator.set_decoherence_D_matrix(D_matrix_eV=D, n=n_case["n"], E0_eV=E0_eV)

                    # Either model this model, or instead build an envelope
                    if not plot_envelope :
                        _plot(color=color, linestyle="-", lw=1, zorder=4, alpha=0.1)

                    else :
                        # Just get osc probs, for calculating envelope
                        osc_probs_E = calculator.calc_osc_prob(
                            initial_flavor=initial_flavor,
                            nubar=nubar, 
                            energy_GeV=E_GeV_values,
                            coszen=coszen_ref,
                        )
                        osc_probs_E = osc_probs_E[:,0,final_flavor]
                        assert osc_probs_E.ndim == 1
                        osc_probs_cz = calculator.calc_osc_prob(
                            initial_flavor=initial_flavor,
                            nubar=nubar, 
                            energy_GeV=n_case["E_GeV_ref"],
                            coszen=coszen_values,
                        )
                        osc_probs_cz = osc_probs_cz[0,:,final_flavor]
                        assert osc_probs_cz.ndim == 1

                        if False : # Only for debugging envelope
                            ax[0].plot( coszen_values, osc_probs_cz, color=color, alpha=1, linestyle="--", zorder=3)
                            ax[1].plot( E_GeV_values, osc_probs_E, color=color, alpha=1, linestyle="--", zorder=3)

                        # Now update envelope
                        if ymax_E is None : # First time
                            ymax_E, ymin_E = osc_probs_E, osc_probs_E
                            ymax_cz, ymin_cz = osc_probs_cz, osc_probs_cz
                        else :
                            ymax_E, ymin_E = np.maximum(ymax_E, osc_probs_E), np.minimum(ymin_E, osc_probs_E)
                            ymax_cz, ymin_cz = np.maximum(ymax_cz, osc_probs_cz), np.minimum(ymin_cz, osc_probs_cz)

            # Plot envelope
            if plot_envelope :
                # Plot vs coszen
                ax[0].fill_between( coszen_values, ymin_cz, ymax_cz, color=color, alpha=0.2, zorder=3)
                ax[1].fill_between( E_GeV_values, ymin_E, ymax_E, color=color, alpha=0.2, zorder=3)


        # Titles
        ax[0].set_title(r"$\cos(\theta)$ = %0.3g" % (coszen_ref) )
        ax[1].set_title(r"$E_\nu$ = %0.3g GeV" % (n_case["E_GeV_ref"]) )
        fig.suptitle(r"$n$ = %i, $\Gamma_0$ = %0.3g eV" % (n_case["n"], n_case["gamma0_eV"]) )

        # Format
        for this_ax in ax :
            this_ax.legend(fontsize=8)
        fig.tight_layout()

    # Report time (can be slow)
    time_taken = datetime.datetime.now() - start_time
    print("Random D matrix comparisons complete : %i models per case : Took %s" % (num_models, time_taken))




def explore_D_matrix_constraints() :
    '''
    Explore what D matrix element values are actually possible, given the varies constraints
    '''


    #
    # Create model
    # 

    calculator = OscCalculator(
        tool=SOLVER,
        atmospheric=True,
        num_neutrinos=3,
    )

    calculator.set_matter("vacuum")


    #
    # Test all gammas, and project
    #

    # Randomly generate matrices and see what works...

    '''
    Things I have learned by doing this (all assuming diagonal only so far):

      (a) If relaxation params are 0 :
         - g1 = g2, g4 = g5, g6 = g7 pair conditions must be respected    (a1-7 conditions imply g1==g2, g=<=g5, g6==g7)
         - there is no relation between the pairs, e.g. g1,4,6, they can take any value w.r.t. each other
         - If one g pair = 0 , the other two pairs MUST equal each other. this is a conical bound I think (consequence of a7)
      (b) If relaxation params are >0 :
         - g1 = g2, g4 = g5, g6 = g7 pair conditions not longer need to be respected    (a1-7 conditions have more freedom due to non-zero g3/8)
         - There are relations between g1/2/4/5/6/7 and g3/8, but still plenty of freedom
         - There is a clear relation limit relating g3 and g8, which does not allow small g3 and large g8 simultaneously  (a1,2,3 conditions combined => g8 >= 3 * g3)
         - There is a clear relation limit between g1/2 and g8, but not for the g4/5/6/7   => g8 >= 3 * g1/2   (ceomes from a1,2,3 conditions, which only affect g1/2/3/8. g3 not limited do to lack of factor 3 there)
    '''

    N = 10000
    random_state = np.random.RandomState(12345)

    disable_relaxation_params = True
    disable_decoh_params = False
    enforce_12_45_67_pairs = True

    valid_D_matrices, invalid_D_matrices = [], []
    for n in range(N) :

        # Generate random D matrix
        D, valid = generate_random_D_matrix(
            calculator=calculator,
            random_state=random_state, 
            disable_relaxation_params=disable_relaxation_params, 
            disable_decoh_params=disable_decoh_params, 
            enforce_12_45_67_pairs=enforce_12_45_67_pairs,
        )

        # Store
        if valid :
            valid_D_matrices.append( np.diagonal(D).tolist() )
        else :
            invalid_D_matrices.append( np.diagonal(D).tolist() )

    valid_D_matrices = np.array(valid_D_matrices)
    invalid_D_matrices = np.array(invalid_D_matrices)

    # Plot pairs...
    # Loop over pairs of interest
    for (i, j) in [ (1, 2), (4, 5), (6, 7), (1, 4), (1, 6), (4, 6), (3, 8), (1, 3), (1, 8), (4, 3), (4, 8), (6, 3), (6, 8) ] :

        # Scatter plots param pairs
        fig, ax = plt.subplots(figsize=(6,5))
        # if np.shape(invalid_D_matrices)[0] > 0 :
        #     ax.scatter(invalid_D_matrices[:,i], invalid_D_matrices[:,j], color="red", marker=".")
        if np.shape(valid_D_matrices)[0] > 0 :
            ax.scatter(valid_D_matrices[:,i], valid_D_matrices[:,j], color="green", marker=".")

        # Add conditions of interest
        if (i in [1,2,3]) and (j==8) :
            x = np.linspace(0., 1., num=100)
            y = 3.*x
            ax.plot(x, y, color="black", linestyle="--", label=r"$\Gamma_{%i} = 3 \Gamma_{%i}$"%(j,i))
        # if (i in [1,2]) and (j==3) :
        #     x = np.linspace(0., 1., num=100)
        #     y = 3.*x/2.
        #     ax.plot(x, y, color="black", linestyle="--", label=r"$\Gamma_{%i} = 3 \Gamma_{%i} / 2$"%(j,i))

        # Format
        ax.set_xlabel(r"$\Gamma_{%i}$"%i)
        ax.set_ylabel(r"$\Gamma_{%i}$"%j)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()





#
# Main
#

if __name__ == "__main__" :

    #
    # Run plotters
    #

    # reproduce_2306_14778()

    # compare_qg_models_coherence_length()

    # explore_D_matrix_constraints()
    compare_random_D_matrix_textures(num_models=1000)

    # compare_D_matrix_textures()


    #TODO verify damping term implement of nuVBH models matches the full solver version


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
