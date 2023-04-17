'''
Making atmopsheric neutrino decoherence plots for
a decoherence theory paper.

These are "everything in" plots showing all modelled 
physics.

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *


#
# Plot functions
#

def plot_distance_dependence(
    physics_cases,
    num_points=1000,
) :
    '''
    Plot the distance-dependence of the decoherence scenarios tested
    '''

    print("\n>>> Plot distance dependence")

    #
    # Prepare plot
    #

    # Create the fig
    fig = Figure( ny=2, figsize=(FIG_WIDTH, 7) )


    #
    # Create calculator
    #

    calculator = DecoherenceCalculator(
        tool=SOLVER,
        atmospheric=False,
        num_neutrinos=NUM_NEUTRINOS,
    )

    calculator.set_calc_basis("nxn")
    calculator.set_matter("vacuum") #TODO matter option


    #
    # Define physics
    #

    linestyle = "-"

    # Steer oscillations
    L_km = np.linspace(0., 5*EARTH_DIAMETER_km, num=num_points)
    INITIAL_FLAVOR = 1
    FINAL_FLAVOR = 1


    #
    # Plot atmospheric parameter space osc probs
    #

    # Loop over models
    for i, physics_case in enumerate(physics_cases) :

        ax = fig.get_ax(y=i)

        # Grab params
        E_GeV = physics_case["E_ref_GeV"]
        model_kw = physics_case["model_kw"]

        # Loop over physics cases
        for case_dict in cases :

            # Set physics params
            if model_kw is None :
                calculator.set_std_osc()
            else :
                calculator.set_decoherence_model(**model_kw)

            # Calc osc prob
            osc_probs = calculator.calc_osc_prob(
                INITIAL_FLAVOR=INITIAL_FLAVOR,
                energy_GeV=E_GeV,
                distance_km=L_km,
            )

            # Plot osc prob
            L_Earth_diameters = L_km / EARTH_DIAMETER_km 
            ax.plot( L_Earth_diameters, osc_probs[0,:,FINAL_FLAVOR], color=case_dict["color"], label=case_dict["label"], linestyle=case_dict["linestyle"] )

        # Overlay limiting cases
        averaged_osc_prob = oscillation_averaged_transition_probability( pmns=calculator.PMNS, INITIAL_FLAVOR=INITIAL_FLAVOR, FINAL_FLAVOR=FINAL_FLAVOR )
        ax.axhline( averaged_osc_prob, color=MODEL_COLORS["randomize_phase"], label="Averaged oscillations", alpha=0.2, linestyle="-", zorder=-1 )
        ax.axhline( 1./float(calculator.num_neutrinos), color=MODEL_COLORS["randomize_state"], label="Equally populated flavors", alpha=0.2, linestyle="-", zorder=-1 )

        # Format figure
        format_ax(ax=ax, xlabel=r"$L / %s$"%EARTH_DIAMETER_TEX, ylabel=r"$%s$"%calculator.get_transition_prob_tex(INITIAL_FLAVOR,FINAL_FLAVOR), legend=False, xlim=(L_Earth_diameters[0],L_Earth_diameters[-1]), ylim=(0.,1.) )


    #
    # Final format
    #

    fig.get_ax(y=0).legend( loc="upper right", fontsize=10, ncol=2 )
    fig.quick_format( legend=False, ylim=(0.,1.) )

    fig.save("atmo_decoherence_vs_distance.pdf")


def plot_oscillograms(
    physics_cases,
    grid_dim=100,
    matter=True,
    titles=True,
) :

    print("\n>>> Plot atmospheric oscillograms")


    #
    # Define physics
    #

    # Get scan grid
    energy_values_GeV = np.logspace(np.log10(ENERGY_RANGE_GeV[0]), np.log10(ENERGY_RANGE_GeV[1]), num=grid_dim)
    coszen_values = np.linspace(COSZEN_RANGE[0], COSZEN_RANGE[1], num=grid_dim)

    # Special handling for nuSQuIDS
    calc_kw = {}
    if SOLVER == "nusquids" :
        calc_kw["interactions"] = matter # Include Earth absorption
        calc_kw["energy_nodes_GeV"] = energy_values_GeV #TODO Seems to be much faster when settings nodes to the same values I later evaluate. Does this mean it is the interpolation that is taking the time?
        calc_kw["coszen_nodes"] = coszen_values 

    # Create calculator
    calculator = DecoherenceCalculator(
        tool=SOLVER,
        atmospheric=True,
        num_neutrinos=NUM_NEUTRINOS,
        **calc_kw
    )

    # Choose matter vs vacuum
    calculator.set_matter("earth" if matter else "vacuum")


    #
    # Calc std oscillogram
    #

    # Disable decoherence
    calculator.set_std_osc()

    # Define args
    plot_oscillogram_kw = dict(
        initial_flavor=INITIAL_FLAVOR,
        final_flavor=FINAL_FLAVOR,
        # initial_rho=INITIAL_RHO,
        energy_GeV=energy_values_GeV,
        coszen=coszen_values, 
    )

    # Plot oscillogram
    fig = Figure( figsize=(FIG_WIDTH, 5), title=( "Standard osc" if titles else None) )
    calculator.plot_oscillogram(ax=fig.get_ax(), **plot_oscillogram_kw)
    for ax in  fig.get_all_ax() :
        ax.set_xscale("log")
    fig.tight_layout()
    fig_name = "atmo_oscillogram_std_osc_%s" % ("matter" if matter else "vacuum")
    fig.save(fig_name+".pdf")
    fig.save(fig_name+".png", dpi=500)


    #
    # Loop over cases
    #

    # Loop over energy dependence cases
    for physics_case in physics_cases :

        # Grab params
        model_kw = physics_case["model_kw"]
        model_name = model_kw["model_name"]
        n = model_kw["n"]


        #
        # Calc decoherence oscillograms
        #

        print( "Computing `%s` n=%i decoherence..." % (model_name, n) )

        # Enable decoherence
        calculator.set_decoherence_model(**model_kw)

        # Plot oscillogram
        fig = Figure( ny=2, figsize=(FIG_WIDTH, 10), title=( "%s, n=%i"%(model_name.replace("_"," ").title(), n) if titles else None) )
        calculator.plot_oscillogram(ax=fig.get_ax(y=0), diff_ax=fig.get_ax(y=1), **plot_oscillogram_kw)

        # Annotate
        if model_name == "randomize_state" :
            fig.get_ax(y=0).annotate("Oscillations", xy=(8., 0.), xytext=(4., 0.5), ha="left", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")
            if np.isclose(n, 0.) :
                if matter :
                    fig.get_ax(y=0).annotate("Earth absorption", xy=(0.7e5, -0.95), xytext=(7e4, 0.6), ha="right", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")
                    # fig.get_ax(y=0).annotate("Earth absorption", xy=(0.6e6, -0.88), xytext=(7e4, 0.6), ha="right", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")
                fig.get_ax(y=0).annotate("Decoherence", xy=(7e2, -0.6), xytext=(1e2, 0.15), ha="left", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")
            elif np.isclose(n, 2.) :
                if matter :
                    fig.get_ax(y=0).annotate("Earth absorption", xy=(0.5e5, -0.9), xytext=(9e2, 0.15), ha="right", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")
                    # fig.get_ax(y=0).annotate("Earth absorption", xy=(0.6e6, -0.9), xytext=(9e2, 0.15), ha="right", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")
                # fig.get_ax(y=0).annotate("Decoherence", xy=(5e3, 0.12), xytext=(2e3, 0.75), ha="right", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")
                fig.get_ax(y=0).annotate("Decoherence", xy=(5e4, 0.12), xytext=(2e3, 0.75), ha="right", va="bottom", arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=10, zorder=9,  xycoords="data")

        # Format
        for ax in  fig.get_all_ax() :
            ax.set_xscale("log")
        fig.tight_layout()
        fig_name = "atmo_oscillogram_%s_n%i_%s" % (model_name, n, "matter" if matter else "vacuum")
        fig.save(fig_name+".pdf")
        fig.save(fig_name+".png", dpi=500)
        


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
    print(D_matrix_eV)

    # Configure decoherence
    calculator.set_decoherence_D_matrix(
        D_matrix_eV=D_matrix_eV,
        n=1, # Energy-dependence
        E0_eV=1.e9, # Reference energy, 1 GeV
    ) 


    #
    # Plot oscillations
    #

    # Define neutrino
    initial_flavor = 1 # muon neutrino
    E_GeV = 100.
    distance_km = np.linspace(0., EARTH_DIAMETER_km, num=100)
    coszen = np.linspace(-1., +1., num=100)

    # Calc osc probs vs distance
    calculator.plot_osc_prob_vs_distance(initial_flavor=initial_flavor, energy_GeV=E_GeV, coszen=coszen, color="orange")

    # Plot 2D oscillogram


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
