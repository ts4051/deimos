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

from matplotlib import cm

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import calc_effective_osc_params_in_matter_2flav


#
# Globals
#

COLORS = [ "red", "blue", "green", "orange", "purple", "magenta" ]


#
# Plotting functions
#

def plot_matter_effects_2flav(solver) :
    '''
    Plotting matter effects in a 2flavor system, for easy comparison with analytic functions

    Useful reference: [1] 
    '''

    if solver == "nusquids" :
        print("nuSQuIDS does not support 2-neutrino systems, skipping...")
        return

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
    
    # Create calculator
    calculator = OscCalculator(
        solver=solver,
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


def plot_high_energy_earth_interaction_effects(solver) :
    '''
    Plotting Earth interaction effects, e.g. absorption and tau/NC regeneration

    Doing this by propagating a flux (regeneration effects are flux-dependent, since neutrino energy changes)
    '''

    if solver == "deimos" :
        print("Earth interactions not implemented in DEIMOS, skipping...")
        return


    #
    # Define parameter space
    #

    num_scan_points = 25
    E_values_GeV = np.geomspace(100., 1e5, num=num_scan_points) # Staying above the standard oscillations for simplicity here
    coszen_values = np.array([-1.,]) # Just testing up-going currently

    nubar = False


    #
    # Define cases
    #

    cases = collections.OrderedDict()
    cases["No interactions"] = {"interactions":False}
    cases["Include interactions/regeneration"] = {"interactions":True}

 
    # 
    # Plot
    #

    # Loop over cases
    for i_case, (case_label, case_kw) in enumerate(cases.items()) :


        #
        # Create model
        #

        calc_kw = {}
        if solver == "nusquids" :
            calc_kw["energy_nodes_GeV"] = E_values_GeV

        # Create calculator
        calculator = OscCalculator(
            solver=solver,
            atmospheric=True,
            **calc_kw,
            **case_kw # This passes the interaction information to the model
        )

        # Enable Earth matter
        calculator.set_matter("earth")


        #
        # Propagate an astrophysical flux
        #

        # Note that the flux has an impact of tau/NC regeneration, since there are HE->LE transitions and so the relative rates of HE to LE neutrinos matters

        # Propagate the atro flux
        initial_flux, final_flux = calculator.calc_final_flux(
            source="astro",
            energy_GeV=E_values_GeV,
            coszen=coszen_values,
            nubar=nubar,
        )

        # Make figure
        fig, ax = plt.subplots( figsize=(6,4) )
        fig.suptitle(case_label)

        # Plot steering
        linestyles = ["-","--", ":"]

        # Loop over flavors
        for i_f in range(calculator.num_neutrinos) :

            # Get the flux for this flavor. Only a single coszen value.
            assert coszen_values.size == 1
            flavor_initial_flux = initial_flux[:,0,i_f]
            flavor_final_flux = final_flux[:,0,i_f]
            ratio = flavor_final_flux / flavor_initial_flux
            assert ratio.ndim == 1

            # Plot ratio vs energy
            ax.plot(E_values_GeV, ratio, color=NU_COLORS[i_f], linestyle=linestyles[i_f], lw=4, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
                       
        # Format
        ax.set_xscale("log")
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_ylabel(r"$\phi_f / \phi_i$")
        ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
        ax.set_ylim(0., 1.1)
        ax.grid(True)
        ax.legend(fontsize=12)
        fig.tight_layout()



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
    args = parser.parse_args()


    #
    # Run plotting functions
    #

    plot_matter_effects_2flav(solver=args.solver)

    plot_high_energy_earth_interaction_effects(solver=args.solver)

    #TODO plot resonance condition
    #TODO 3 flavor nu and nubar matter plots
    #TODO low energy atmospheric matter effects (e.g. NMO signal)

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
