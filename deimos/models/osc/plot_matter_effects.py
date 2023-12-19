'''
Investigating oscillation matter effects in simple settings

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *


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
# PLotting functions
#

def plot_matter_effects_2flav() :
    '''
    Plotting matter effects in a 2flavor system, for easy comparison with analytic functions

    Useful reference: [1] https://www.hindawi.com/journals/ahep/2013/972485/
    '''



    #
    # Define neutrino
    #

    # Oscillations
    flavors = ["e", "mu"] # 2 flavors for simplicity
    mixing_angles_rad = [np.deg2rad(45.)]
    mass_splittings_eV2 = [2.5e-3]
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
        mixing_angles_rad=mixing_angles_rad,
        mass_splittings_eV2=mass_splittings_eV2,
        deltacp_rad=deltacp_rad,
    )


    #
    # Loop over matter cases
    #

    # Define cases
    cases = collections.OrderedDict()
    cases["Vacuum"] = {"matter":"vacuum"}
    for rho in [ 1., 5., 10., 100., 1000. ] :
        cases[r"$\rho$ = %i"%rho] = {"matter":"constant", "matter_density_g_per_cm3":rho, "electron_fraction":0.5}

    # Loop over cases
    fig, ax = None, None
    for case_label, case_kw in cases.items() :

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
        )


        # Calculate effective oscillation parameters in the matter potential


        #TODO resonance density  - see equation 24


    # Format figure




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

        case_colors = [ "red", "blue", "green", "orange", "purple", "magenta" ]

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
                        color=adjust_lightness(case_colors[i_case], color_adjust),
                        lw=4,
                        # color=color_prefix+case_colors[i_case], 
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



#
# Main
#

if __name__ == "__main__" :

    # plot_matter_effects_2flav()
    
    compare_matter_effects_between_solvers()

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
