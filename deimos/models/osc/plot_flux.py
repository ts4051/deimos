'''
Basic plots of fluxes included in the osc calculator

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

    plot_2D = True

    E_values_GeV = np.geomspace(1., 1e5, num=100)
    coszen_values = np.linspace(-1., +1., num=100)

    nubar = False


    #
    # Create model
    #

    solver = "nusquids"
    
    # Tool specific configuration
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_values_GeV
        kw["interactions"] = True

    # Create calculator
    calculator = OscCalculator(
        tool=solver,
        atmospheric=True,
        **kw
    )

    # Use Earth model
    calculator.set_matter("earth")


    #
    # Define cases
    #

    cases = collections.OrderedDict()
    cases["Atmo"] = ("atmo", "mceq", 3.)
    cases["Astro"] = ("astro", None, 2.)

    # Loop over cases
    for flux_label, (flux_source, flux_tool, flux_E_power) in cases.items() :


        #
        # Propagate flux
        #

        # Propagate the atmospheric flux
        initial_flux, final_flux = calculator.calc_final_flux(
            source=flux_source,
            tool=flux_tool,
            energy_GeV=E_values_GeV,
            coszen=coszen_values,
            nubar=nubar,
        )


        #
        # Plot initital flux
        #

        # Plot steering
        E_color_norm = matplotlib.colors.Normalize(vmin=0, vmax=E_values_GeV.size)
        cz_color_norm = matplotlib.colors.Normalize(vmin=0, vmax=coszen_values.size)

        # Make fig
        nx = 2
        ny = calculator.num_neutrinos
        fig_E, ax_E = plt.subplots( ncols=nx, nrows=ny, figsize=(6*nx, 4*ny) )
        fig_cz, ax_cz = plt.subplots( ncols=nx, nrows=ny, figsize=(6*nx, 4*ny) )
        if plot_2D :
            fig_2D, ax_2D = plt.subplots( ncols=nx, nrows=ny, figsize=(6*nx, 6*ny) )

        # Title
        fig_E.suptitle(flux_label)
        fig_cz.suptitle(flux_label)
        if plot_2D :
            fig_2D.suptitle(flux_label)

        # Loop over flavors
        for i_f in range(calculator.num_neutrinos) :

            # fig.suptitle( r"$%s$"%calculator.get_nu_flavor_tex(i=i_f, nubar=nubar) )

            # Plot each flavor in its own row
            y = i_f

            # Extract flux for this flavor, vs energy (only providing single coszen value here)
            flavor_initial_flux = initial_flux[:,:,i_f]
            flavor_final_flux = final_flux[:,:,i_f]
            # ratio = flavor_final_flux / flavor_initial_flux

            # Plot steering
            initial_flux_tex = r"\phi_{%s,i}" % calculator.get_flavor_tex(i_f)
            final_flux_tex = r"\phi_{%s,f}" % calculator.get_flavor_tex(i_f)

            # [E, coszen] plot
            if plot_2D :
                plot_colormap( ax=ax_2D[y,0], x=E_values_GeV, y=coszen_values, z=flavor_initial_flux, cmap="jet", zlabel=r"$%s$"%initial_flux_tex )
                plot_colormap( ax=ax_2D[y,1], x=E_values_GeV, y=coszen_values, z=flavor_final_flux, cmap="jet", zlabel=r"$%s$"%final_flux_tex )
                # plot_colormap( ax_2D=ax_2D[2,0], x=E_values_GeV, y=coszen_values, z=ratio, cmap="jet", zlabel=r"$\phi / \phi_0$" )
                for x in range(nx) :
                    ax_2D[y,x].set_xscale("log")
                    ax_2D[y,x].set_xlabel(r"$E$ [GeV]")
                    ax_2D[y,x].set_ylabel(r"$\cos(\theta)$")

            # 1D vs energy
            for i_cz, coszen in enumerate(coszen_values) :
                label = r"$\cos(\theta)$ = %0.3g"%(coszen) if i_cz in [0, coszen_values.size-1] else None
                ax_E[y,0].plot( E_values_GeV, np.power(E_values_GeV, flux_E_power)*flavor_initial_flux[:,i_cz], color=cm.jet(cz_color_norm(i_cz)), label=label )
                ax_E[y,1].plot( E_values_GeV, np.power(E_values_GeV, flux_E_power)*flavor_final_flux[:,i_cz], color=cm.jet(cz_color_norm(i_cz)), label=label )
                # ax_E[2,x].plot( E_values_GeV, ratio[:,i_cz], color=cm.jet(cz_color_norm(i_cz)), label=r"$\cos(\theta)$ = %0.3g"%(coszen,) )
            for x in range(nx) :
                ax_E[y,x].set_xscale("log")
                ax_E[y,x].set_xlim(E_values_GeV[0], E_values_GeV[-1])
                ax_E[y,x].set_ylim(0., None)
                ax_E[y,x].legend(fontsize=8)
                ax_E[y,x].grid(True)
                ax_E[-1, x].set_xlabel(r"$E$ [GeV]")
            ax_E[y,0].set_ylabel(r"$E^{%0.3g} %s$"%(flux_E_power, initial_flux_tex))
            ax_E[y,1].set_ylabel(r"$E^{%0.3g} %s$"%(flux_E_power, final_flux_tex))
            # ax_E[2,1].set_ylabel(r"\phi / \phi_0$")

            # 1D vs coszen
            for i_E, E_GeV in enumerate(E_values_GeV) :
                label = r"$E$ = %0.3g GeV"%(E_GeV,) if i_E in [0, E_values_GeV.size-1] else None
                ax_cz[y,0].plot( coszen_values, flavor_initial_flux[i_E,:]/np.max(flavor_initial_flux[i_E,:]), color=cm.jet(E_color_norm(i_E)), label=label )
                ax_cz[y,1].plot( coszen_values, flavor_initial_flux[i_E,:]/np.max(flavor_initial_flux[i_E,:]), color=cm.jet(E_color_norm(i_E)), label=label )
                # ax_cz[2,2].plot( coszen_values, ratio[i_E,:], color=cm.jet(E_color_norm(i_E)), label=r"$E$ = %0.3g GeV"%(E_GeV,) )
            for x in range(nx) :
                ax_cz[y,x].set_xlim(coszen_values[0], coszen_values[-1])
                ax_cz[y,x].set_ylim(0., None)
                ax_cz[y,x].grid(True)
                ax_cz[y,x].legend(fontsize=8)
                ax_cz[-1,x].set_xlabel(r"$\cos(\theta)$")
            ax_cz[y,0].set_ylabel(r"$%s$ (norm.)"%initial_flux_tex)
            ax_cz[y,1].set_ylabel(r"$%s$ (norm.)"%final_flux_tex)
            # ax_cz[0,2].set_ylabel(r"$\phi / \phi_0$")

    # Format
    # fig_E.subplots_adjust(wspace=0, hspace=0)
    fig_E.tight_layout()
    # fig_cz.subplots_adjust(wspace=0, hspace=0)
    fig_cz.tight_layout()
    if plot_2D :
        # fig_2D.subplots_adjust(wspace=0, hspace=0)
        fig_2D.tight_layout()


    # #
    # # Plot propagated flux
    # #

    # initial_state = np.full( (coszen_values.size, E_values_GeV.size, calculator.num_neutrinos), np.NaN )

    # for (flavor, nubar), flux in initial_flux.items() :

    #     if nubar == False : #TODO

    

    #         # Determine shape of initial state vector in nuSQuIDS
    #         state_shape = [ self.nusquids.GetNumCos(), self.nusquids.GetNumE() ]
    #         state_shape.append( 2 ) # nu/nubar
    #         state_shape.append( final_flavors.size )
    #         state_shape = tuple(state_shape)

    #         # Define initial state if not provided, otherwise verify the one provided
    #         if initial_state is None :
    #             initial_state = np.full( state_shape, 0. )
    #             initial_state[ :, :, rho, initial_flavor ] = 1. # dims = [ cz node, E node, nu(bar), flavor ]
    #         else :
    #             assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

    #         # Set the intial





    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
