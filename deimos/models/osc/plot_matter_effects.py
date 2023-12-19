'''
Investigating oscillation matter effects in simple settings

Useful refs:
  [1] https://www.hindawi.com/journals/ahep/2013/972485/
  [2] https://arxiv.org/pdf/hep-ph/0412391.pdf
  [3] https://cds.cern.ch/record/1114392/files/p159.pdf
  [4] https://arxiv.org/abs/1806.11051

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import calc_effective_osc_params_in_matter_2flav


#
# Globals
#

COLORS = [ "red", "blue", "green", "orange", "purple", "magenta" ]


#
# Helper functions
#

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


#
# Plotting functions
#

def plot_matter_effects_2flav() :
    '''
    Plotting matter effects in a 2flavor system, for easy comparison with analytic functions

    Useful reference: [1] 
    '''



    #
    # Define neutrino
    #

    # Oscillations
    flavors = ["e", "mu"] # 2 flavors for simplicity
    mixing_angle_rad = np.deg2rad(45.)
    mass_splitting_eV2 = 2.5e-3
    deltacp_rad = 0.

    # Neutrino
    initial_flavor, final_flavor, nubar = 1, 0, False # numu->nue
    E_GeV = 1.
    L_km = np.linspace(0., 1000., num=100)


    #
    # Create model
    #
    
    solver = "deimos"

    # Create calculator
    calculator = OscCalculator(
        tool=solver,
        atmospheric=False,
        flavors=flavors,
        mixing_angles_rad=[mixing_angle_rad],
        mass_splittings_eV2=[mass_splitting_eV2],
        deltacp_rad=deltacp_rad,
    )


    #
    # Loop over matter cases
    #

    # Define cases
    cases = collections.OrderedDict()
    cases["Vacuum"] = {"matter":"vacuum"}
    for rho in [ 5., 10., 100., 1000. ] :
        cases[r"$\rho$ = %i"%rho] = {"matter":"constant", "matter_density_g_per_cm3":rho, "electron_fraction":0.5}

    # Loop over cases
    fig, ax = None, None
    for i_case, (case_label, case_kw) in enumerate(cases.items()) :

        # Set matter
        calculator.set_matter(**case_kw)

        # Calc osc probs
        fig, ax, _ = calculator.plot_osc_prob_vs_distance(
            fig=fig,
            ax=ax,
            initial_flavor=initial_flavor, 
            final_flavor=final_flavor, 
            nubar=nubar,
            energy_GeV=E_GeV,
            distance_km=L_km,
            label=case_label, 
            color=adjust_lightness(COLORS[i_case], 0.8),
            lw=4,
        )

        # Next stuff is matter-specific
        if case_kw["matter"] != "vacuum" :

            # Calculate effective oscillation parameters in the matter potential
            matter_mixing_angle_rad, matter_mass_splitting_eV2 = calc_effective_osc_params_in_matter_2flav(
                E_eV=E_GeV*1e9, 
                mixing_angle_rad=mixing_angle_rad, 
                mass_splitting_eV2=mass_splitting_eV2, 
                matter_density_g_per_cm3=case_kw["matter_density_g_per_cm3"], 
                electron_fraction=case_kw["electron_fraction"],
            )

            # Calc vacuum oscillations but using the effect matter osc params
            calculator.set_matter("vacuum")
            calculator.set_mixing_angles(matter_mixing_angle_rad, deltacp=deltacp_rad)
            calculator.set_mass_splittings(matter_mass_splitting_eV2)

            fig, ax, _ = calculator.plot_osc_prob_vs_distance(
                fig=fig,
                ax=ax,
                initial_flavor=initial_flavor, 
                final_flavor=final_flavor, 
                nubar=nubar,
                energy_GeV=E_GeV,
                distance_km=L_km,
                # label=case_label, 
                linestyle=":",
                color=adjust_lightness(COLORS[i_case], 1.5),
                lw=4,
            )

            # Now reset to original vacuum mixing angles
            calculator.set_mixing_angles(mixing_angle_rad, deltacp=deltacp_rad)
            calculator.set_mass_splittings(mass_splitting_eV2)




def compare_matter_effects_between_solvers() :
    '''
    Cross-check matter effects between solvers
    '''

    #
    # Define neutrino
    #

    # Neutrino
    initial_flavor = 1
    final_flavor_values = [0, 1, 2]
    nubar_values = [False, True] 
    E_GeV = 1.
    L_km = np.linspace(0., 1000., num=100)



    #
    # Init figure
    #

    nx, ny = len(nubar_values), len(final_flavor_values)
    fig, ax = plt.subplots( ncols=nx, nrows=ny, figsize=(6*nx, 3*ny) )


    #
    # Loop over solvers
    #

    solvers = [ "deimos", "nusquids", "prob3" ]
    linestyles = [ "-", "--", ":" ]
    color_adjustments = [ 0.7, 1., 1.5]

    # for solver, linestyle, alpha, color_prefix in zip(solvers, linestyles, alpha, color_prefixes) :
    for solver, linestyle, color_adjust in zip(solvers, linestyles, color_adjustments) :


        #
        # Create model
        #
        
        # Tool specific configuration
        kw = {}
        if solver == "nusquids" :
            # kw["energy_nodes_GeV"] = False # Single energy mode
            kw["energy_nodes_GeV"] = np.array([ 0.5*E_GeV, E_GeV, 2.*E_GeV ]) # Approximating single energy mode since having issues with the real one
            kw["interactions"] = False # Not allowed in single energy mode

        # Create calculator
        calculator = OscCalculator(
            tool=solver,
            atmospheric=False,
            **kw
        )


        #
        # Loop over matter cases
        #

        # Define cases
        cases = collections.OrderedDict()
        cases["Vacuum"] = {"matter":"vacuum"}
        for rho in [ 5., 100., 1000. ] :
            cases[r"$\rho$ = %i g/cm$^3$"%rho] = {"matter":"constant", "matter_density_g_per_cm3":rho, "electron_fraction":0.5}

        # Loop over cases
        for i_case, (case_label, case_kw) in enumerate( cases.items() ) :

            # Set matter
            calculator.set_matter(**case_kw)

            # Loop over nu/nubar
            for x, nubar in enumerate(nubar_values) :

                # Calc osc probs
                osc_probs = calculator.calc_osc_prob(
                    initial_flavor=initial_flavor, 
                    nubar=nubar,
                    energy_GeV=E_GeV,
                    distance_km=L_km,
                )

                # Loop over final flavors
                for y, final_flavor in enumerate(final_flavor_values) :

                    # Plot osc probs
                    ax[y,x].plot(
                        L_km, 
                        osc_probs[:,final_flavor], 
                        label="%s, %s"%(solver, case_label), 
                        linestyle=linestyle,   
                        color=adjust_lightness(COLORS[i_case], color_adjust),
                        lw=4,
                        # color=color_prefix+COLORS[i_case], 
                        # alpha=alpha,
                    )

                    # Format ax
                    ax[y,x].set_xlim(L_km[0], L_km[-1])
                    if final_flavor == 0 :
                        ax[y,x].set_ylim(0., 0.2)
                    else :
                        ax[y,x].set_ylim(0., 1.)
                    ax[y,x].set_xlabel(DISTANCE_LABEL)
                    ax[y,x].set_ylabel( r"$%s$"%calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )
                    ax[y,x].grid(True)
                    if (y==0) and (x==0) :
                        ax[y,x].legend(ncol=len(solvers), fontsize=6, loc="upper center")

    # Format figure
    fig.tight_layout()



def verify_matter_layers_implementation() :
    '''
    Perform checks that the "layers" mode for matter is working
    '''

    #
    # Define neutrino
    #

    # Neutrino
    initial_flavor = 1
    final_flavor_values = [0, 1, 2]
    nubar_values = [False, True] 
    E_GeV = 1.
    L_km = np.linspace(0., 1000., num=100)


    #
    # Define test cases
    #

    cases = collections.OrderedDict()

    cases["Vacuum vs const"] = ( {"matter":"vacuum"}, {"matter":"constant",  "matter_density_g_per_cm3":0., "electron_fraction":0. } )

    layer_endpoints = np.linspace(0., L_km[-1], num=11)[1:]
    cases["Vacuum vs layers"] = ( 
        {"matter":"vacuum"},
        {"matter":"layers", "layer_endpoint_km":layer_endpoints, "matter_density_g_per_cm3":np.zeros_like(layer_endpoints), "electron_fraction":np.zeros_like(layer_endpoints) },
    )

    matter_density_g_per_cm3, electron_fraction = 10., 0.5
    cases["Const vs layers"] = ( 
        {"matter":"constant", "matter_density_g_per_cm3":matter_density_g_per_cm3, "electron_fraction":electron_fraction },
        {"matter":"layers", "layer_endpoint_km":layer_endpoints, "matter_density_g_per_cm3":np.full_like(layer_endpoints, matter_density_g_per_cm3), "electron_fraction":np.full_like(layer_endpoints, electron_fraction)}, 
     )

    layer_endpoint_too_long = np.linspace(0., L_km[-1]*5., num=100)[1:]
    cases["Layers extending past L"] = ( 
        {"matter":"layers", "layer_endpoint_km":layer_endpoints, "matter_density_g_per_cm3":np.zeros_like(layer_endpoints), "electron_fraction":np.zeros_like(layer_endpoints) },
        {"matter":"layers", "layer_endpoint_km":layer_endpoint_too_long, "matter_density_g_per_cm3":np.zeros_like(layer_endpoint_too_long), "electron_fraction":np.zeros_like(layer_endpoint_too_long) },
    )

    cases["Differing layers but same profile"] = ( 
        {"matter":"layers", "layer_endpoint_km":np.array([500., 750., 1000.]), "matter_density_g_per_cm3":np.array([5., 10., 20.]), "electron_fraction":np.array([0.5, 0.5, 0.5]) },
        {"matter":"layers", "layer_endpoint_km":np.array([250., 500., 750., 770., 950., 1000., 1200.]), "matter_density_g_per_cm3":np.array([5., 5., 10., 20., 20., 20., 30.]), "electron_fraction":np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) },
    )


    #
    # Loop over test cases
    #

    for i_case, ( case_label, (ref_case_kw, test_case_kw) ) in enumerate( cases.items() ) :


        #
        # Init figure
        #

        nx, ny = len(nubar_values), len(final_flavor_values)
        fig, ax = plt.subplots( ncols=nx, nrows=ny, figsize=(6*nx, 3*ny) )

        fig.suptitle(case_label)


        #
        # Create model
        #

        solver = "nusquids" # 
        
        # Tool specific configuration
        kw = {}
        if solver == "nusquids" :
            kw["energy_nodes_GeV"] = np.array([ 0.5*E_GeV, E_GeV, 2.*E_GeV ]) # Approximating single energy mode since having issues with the real one
            kw["interactions"] = False # Not allowed in single energy mode

        # Create calculator
        calculator = OscCalculator(
            tool=solver,
            atmospheric=False,
            **kw
        )

        # Loop over nu/nubar
        for x, nubar in enumerate(nubar_values) :

            # Loop over ref/test cases
            for case_kw, color, linestyle in zip([ref_case_kw, test_case_kw], ["red", "blue"], ["-", ":"]) :

                # Set matter
                calculator.set_matter(**case_kw)

                # Calc osc probs
                osc_probs = calculator.calc_osc_prob(
                    initial_flavor=initial_flavor, 
                    nubar=nubar,
                    energy_GeV=E_GeV,
                    distance_km=L_km,
                )

                # Loop over final flavors
                for y, final_flavor in enumerate(final_flavor_values) :

                    # Draw layers
                    if case_kw["matter"] == "layers" :
                        for l in case_kw["layer_endpoint_km"] :
                            ax[y,x].axvline(l, color=color, linestyle=linestyle, lw=2, alpha=0.7, zorder=5)

                    # Plot osc probs
                    ax[y,x].plot(
                        L_km, 
                        osc_probs[:,final_flavor], 
                        linestyle=linestyle,   
                        color=color,
                        lw=4,
                        zorder=6,
                    )

                    # Format ax
                    ax[y,x].set_xlim(L_km[0], L_km[-1])
                    if final_flavor == 0 :
                        ax[y,x].set_ylim(0., 0.2)
                    else :
                        ax[y,x].set_ylim(0., 1.)
                    ax[y,x].set_xlabel(DISTANCE_LABEL)
                    ax[y,x].set_ylabel( r"$%s$"%calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )
                    ax[y,x].grid(True)

        # Format figure
        fig.tight_layout()




#
# Main
#

if __name__ == "__main__" :

    plot_matter_effects_2flav()

    compare_matter_effects_between_solvers()

    #TODO plot resonance condition

    verify_matter_layers_implementation()

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
