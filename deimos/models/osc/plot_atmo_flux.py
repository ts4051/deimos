'''
Basic plots of atmospheric neutrino flux with oscillations and Earth absorption taken into account

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
    # Parameter space
    #

    E_values_GeV = np.geomspace(1., 1e5, num=50)
    coszen_values = np.linspace(-1., +1., num=40)


    #
    # Create model
    #

    solver = "deimos"
    
    # Tool specific configuration
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_values_GeV

    # Create calculator
    calculator = OscCalculator(
        tool=solver,
        atmospheric=True,
        **kw
    )

    # Use Earth model
    # calculator.set_matter("earth")   #TODO
    calculator.set_matter("vacuum")


    #
    # Get initial flux
    #
    
    initial_flux = calculator.get_atmospheric_neutrino_flux(energy_GeV=E_values_GeV, coszen=coszen_values, grid=True, overwrite_cache=False)


    #
    # Plot initital flux
    #

    # Plot steering
    E_power = 3.
    E_color_norm = matplotlib.colors.Normalize(vmin=0, vmax=E_values_GeV.size)
    cz_color_norm = matplotlib.colors.Normalize(vmin=0, vmax=coszen_values.size)

    # Loop over flaors
    for (flavor, nubar), flux in initial_flux.items() :

        fig, ax = plt.subplots( ncols=3, figsize=(18,6) )
        fig.suptitle( r"$%s$"%calculator.get_nu_flavor_tex(i=flavor, nubar=nubar) )
 
        # 2D plot
        plot_colormap( ax=ax[0], x=E_values_GeV, y=coszen_values, z=flux, cmap="jet", zlabel=r"$\phi$" )
        ax[0].set_xscale("log")
        ax[0].set_xlabel(r"$E$ [GeV]")
        ax[0].set_ylabel(r"$\cos(\theta)$")

        # 1D vs energy
        for i_cz, coszen in enumerate(coszen_values) :
            ax[1].plot( E_values_GeV, np.power(E_values_GeV, E_power)*flux[:,i_cz], color=cm.jet(cz_color_norm(i_cz)), label=r"$\cos(\theta)$ = %0.3g"%(coszen,) )
        ax[1].legend(fontsize=8, ncol=2)
        ax[1].set_xscale("log")
        ax[1].set_xlabel(r"$E$ [GeV]")
        ax[1].set_ylabel(r"$E^{%0.3g} \phi$"%(E_power))

        # 1D vs coszen
        for i_E, E_GeV in enumerate(E_values_GeV) :
            ax[2].plot( coszen_values, flux[i_E,:]/np.max(flux[i_E,:]), color=cm.jet(E_color_norm(i_E)), label=r"$E$ = %0.3g GeV"%(E_GeV,) )
        ax[2].legend(fontsize=8, ncol=2)
        ax[2].set_xlabel(r"$\cos(\theta)$")
        ax[2].set_ylabel(r"$\phi$ (norm.)")

        fig.tight_layout()


    #
    # Plot propagated flux
    #

    #TODO



    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
