'''
Plot the astrophysical neutrino flux genrated by OscCalculator, 
both pre- and post-propagation through the Earth.

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
    coszen_values = np.array([-1., 0., +1.]) # upgoing, horizon, downgoing

    matter = "earth"

    solver = "nusquids"


    #
    # Create model
    #
    
    # Tool specific configuration
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_values_GeV
        kw["coszen_nodes"] = coszen_values
        kw["interactions"] = True

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
        **kw
    )

    # Set matter
    calculator.set_matter(matter)


    #
    # Create and propagate the flux
    #

    initial_flux = calculator.get_neutrino_flux(
        source="astro",
        energy_GeV=E_values_GeV,
        coszen=coszen_values,
    )

    final_flux = calculator.propagate_flux(
        initial_flux=initial_flux,
        energy_GeV=E_values_GeV,
        coszen=coszen_values,
    )


    #
    # Plot flux vs energy
    #

    # Choose energy power to multiple flux by when plotting
    energy_power = 2.

    # Loop over flavors
    for i_f in range(calculator.num_neutrinos) :
        for i_nubar, nubar in enumerate([False, True]) :

            # Make fig
            fig, ax = plt.subplots( nrows=2, ncols=len(coszen_values), figsize=(4*len(coszen_values), 5) )
            fig.suptitle( r"$%s$" % (calculator.get_nu_flavor_tex(i_f, nubar)))

            # Loop over coszen values
            for i_cz, coszen in enumerate(coszen_values) :

                # Label the coszen value
                ax[0, i_cz].set_title( r"coszen = %0.3g" % (coszen) )

                # Extract flux for this flavor
                flavor_initial_flux = initial_flux[:, i_cz, i_f, i_nubar]
                flavor_final_flux = final_flux[:, i_cz, i_f, i_nubar]

                # Ratios
                ratio = flavor_final_flux / flavor_initial_flux

                # Plot flux vs energy
                ax[0, i_cz].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_initial_flux, color="black", linestyle="-", lw=3, label="Initial flux" )
                ax[0, i_cz].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_final_flux, color="orange", linestyle="--", lw=3, label="Final flux" )

                # Plot ratios
                ax[1, i_cz].axhline( 1., color="grey", linestyle="-", lw=3)
                ax[1, i_cz].plot( E_values_GeV, ratio, color="orange", linestyle="--", lw=3)

                # Format
                for row in range(2) :
                    ax[row, i_cz].set_xscale("log")
                    ax[row, i_cz].set_xlim(E_values_GeV[0], E_values_GeV[-1])
                    ax[row, i_cz].grid(True)
                ax[0, i_cz].legend(fontsize=8)
                ax[0, i_cz].set_ylim(0., None)
                ax[1, i_cz].set_ylim(0., 1.2)
                ax[1, i_cz].set_xlabel(r"$E$ [GeV]")
                ax[0, i_cz].set_ylabel(  r"$" + (r"" if energy_power == 0. else r"E^{%0.3g}"%energy_power) + r"\phi" + r"$" )
                ax[1, i_cz].set_ylabel( "Final/initial flux" )

            # Format
            fig.tight_layout()


    # Save
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
