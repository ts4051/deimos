'''
Plot impact of Earth interactions (absorption, NC regeneration, tau regeneration) on Earth-crossing high energy neutrinos

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *

import matplotlib
from matplotlib import cm


#
# Main
#

if __name__ == "__main__" :



    #
    # Steering
    #

    E_values_GeV = np.geomspace(1., 1e5, num=100)
    coszen_values = np.linspace(-1., +1., num=5)


    #
    # Create models
    #

    init_kw = dict(
        solver="nusquids",
        atmospheric=True,
        energy_nodes_GeV=E_values_GeV,
        coszen_nodes=coszen_values,
        mixing_angles_rad=(0., 0., 0.), # Disable oscillations (make interaction effects clearer)
        matter="earth",
    )

    # No interactions
    calc_no_int = OscCalculator(
        interactions=False,
        **init_kw
    )

    # Interactions but no tau decay
    calc_int = OscCalculator(
        interactions=True,
        **init_kw
    )
    calc_int.nusquids.Set_TauRegeneration(False)

    # Interactions + tau decay
    calc_int_tau = OscCalculator(
        interactions=True,
        **init_kw
    )


    #
    # Define flux
    #

    # Injecting a toy 1:1:1:1:1:1 flux that is flat in energy and zenith
    # This is to make the effect of interactions clearer
    initial_flux = np.ones( (E_values_GeV.size, coszen_values.size, calc_no_int.num_neutrinos, 2) )
    initial_flux_label = "Flat 1:1:1:1:1:1"
    energy_power = 0.

    # Alternatively, inject a realistic flux (atmospheric)
    # initial_flux = calc_no_int.get_neutrino_flux(energy_GeV=E_values_GeV, coszen=coszen_values, source="atmo", model="mceq", grid=True)
    # initial_flux_label = "Atmospheric flux"
    # energy_power = 3.


    #
    # Propagate flux
    #

    propagate_flux_kw = dict(
        initial_flux=initial_flux,
        energy_GeV=E_values_GeV,
        coszen=coszen_values,
    )

    final_flux_no_int = calc_no_int.propagate_flux(**propagate_flux_kw)
    final_flux_int = calc_int.propagate_flux(**propagate_flux_kw)
    final_flux_int_tau = calc_int_tau.propagate_flux(**propagate_flux_kw)


    #
    # Plot
    #

    # Choose coszen to plot
    coszen_idx = 0
    coszen = coszen_values[coszen_idx]

    # Loop over flavors
    for i_f in range(calc_no_int.num_neutrinos) :
        for i_nubar, nubar in enumerate([False, True]) :

            fig, ax = plt.subplots( nrows=2, figsize=(6, 6) )
            fig.suptitle( r"$%s$ (coszen = %0.3g)" % (calc_no_int.get_nu_flavor_tex(i_f, nubar), coszen) )

            # Extract flux for this flavor
            flavor_initial_flux = initial_flux[:, coszen_idx, i_f, i_nubar]
            flavor_final_flux_no_int = final_flux_no_int[:, coszen_idx, i_f, i_nubar]
            flavor_final_flux_int = final_flux_int[:, coszen_idx, i_f, i_nubar]
            flavor_final_flux_int_tau = final_flux_int_tau[:, coszen_idx, i_f, i_nubar]

            # Ratios
            ratio_no_int = flavor_final_flux_no_int / flavor_initial_flux
            ratio_int = flavor_final_flux_int / flavor_initial_flux
            ratio_int_tau = flavor_final_flux_int_tau / flavor_initial_flux

            # Plot flux vs energy
            ax[0].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_initial_flux, color="black", linestyle="-", lw=3, label="Initial flux" )
            ax[0].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_final_flux_no_int, color="yellow", linestyle="--", lw=3, label="Final flux (w/o interactions)" )
            ax[0].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_final_flux_int, color="orange", linestyle="-.", lw=3, label="Final flux (w/ interactions)" )
            ax[0].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_final_flux_int_tau, color="red", linestyle=":", lw=3, label="Final flux (w/ interactions + tau decay)" )

            # Plot ratios
            ax[1].axhline( 1., color="black", linestyle="-", lw=3)
            ax[1].plot( E_values_GeV, ratio_no_int, color="yellow", linestyle="--", lw=3)
            ax[1].plot( E_values_GeV, ratio_int, color="orange", linestyle="-.", lw=3)
            ax[1].plot( E_values_GeV, ratio_int_tau, color="red", linestyle=":", lw=3)

            # Format
            for this_ax in ax :
                this_ax.set_xscale("log")
                this_ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
                # this_ax.set_ylim(0., None)
                this_ax.legend(fontsize=8)
                this_ax.grid(True)
            ax[0].set_ylim(0., None)
            ax[1].set_ylim(0., None)
            ax[-1].set_xlabel(r"$E$ [GeV]")
            ax[0].set_ylabel( "Flux" )
            ax[1].set_ylabel( "Final/initial flux" )

        # Format
        fig.tight_layout()



    # Save
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
