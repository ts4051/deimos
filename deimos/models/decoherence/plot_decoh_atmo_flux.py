'''
Plot the atmospheroc neutrino flux genrated by OscCalculator, 
both pre- and post-propagation through the Earth.

Tom Stuttard
'''

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *

from deimos.models.decoherence.nuVBH_model import get_vVBH_gamma_eV_from_coherence_length_km, convert_gamma_eV_to_gamma_0_eV


#
# Main
#

if __name__ == "__main__" :


    #
    # Steering
    #

    E_values_GeV = np.geomspace(1., 1e6, num=1000)
    coszen_values = np.array([-1., 0., +1.]) # upgoing, horizon, downgoing

    matter = "earth" # earth vacuum

    solver = "nusquids"

    # Configure decoherence model
    E0_eV = 1e9
    n = 0
    gamma_eV = get_vVBH_gamma_eV_from_coherence_length_km(EARTH_DIAMETER_km) # Coherence length = Earth diameter
    gamma0_eV = convert_gamma_eV_to_gamma_0_eV(gamma_eV=gamma_eV, E_eV=E0_eV, E0_eV=E0_eV, n=n)


    #
    # Create model
    #
    
    # Tool specific configuration
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_values_GeV
        kw["coszen_nodes"] = coszen_values
        kw["interactions"] = True
        kw["nusquids_variant"] = "decoherence"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
        matter=matter,
        **kw
    )


    #
    # Create and propagate the flux
    #

    initial_flux = calculator.get_neutrino_flux(
        source="atmo",
        model="mceq",
        grid=True,
        overwrite_cache=False,
        energy_GeV=E_values_GeV,
        coszen=coszen_values,
    )

    calculator.set_std_osc()
    final_flux_std = calculator.propagate_flux(
        initial_flux=initial_flux,
        energy_GeV=E_values_GeV,
        coszen=coszen_values,
    )

    calculator.set_decoherence_model(model_name="randomize_state", gamma0_eV=gamma0_eV, n=n, E0_eV=E0_eV)
    final_flux_decoh = calculator.propagate_flux(
        initial_flux=initial_flux,
        energy_GeV=E_values_GeV,
        coszen=coszen_values,
    )


    #
    # Plot flux vs energy
    #

    # Choose energy power to multiple flux by when plotting
    energy_power = 3.

    # Loop over flavors
    for i_f in range(calculator.num_neutrinos) :
        for i_nubar, nubar in enumerate([False, True]) :

            # Make fig
            fig, ax = plt.subplots( nrows=2, ncols=len(coszen_values), figsize=(4*len(coszen_values), 5), gridspec_kw={'height_ratios': [2., 1.]} )
            fig.suptitle( r"$%s$" % (calculator.get_nu_flavor_tex(i_f, nubar)))

            # Loop over coszen values
            for i_cz, coszen in enumerate(coszen_values) :

                # Label the coszen value
                ax[0, i_cz].set_title( r"coszen = %0.3g" % (coszen) )

                # Extract flux for this flavor
                flavor_initial_flux = initial_flux[:, i_cz, i_f, i_nubar]
                flavor_final_flux_std = final_flux_std[:, i_cz, i_f, i_nubar]
                flavor_final_flux_decoh = final_flux_decoh[:, i_cz, i_f, i_nubar]

                # Ratios
                ratio_std = flavor_final_flux_std / flavor_initial_flux
                ratio_decoh = flavor_final_flux_decoh / flavor_initial_flux

                # Plot flux vs energy
                ax[0, i_cz].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_initial_flux, color="black", linestyle="-", lw=3, label="Initial flux" )
                ax[0, i_cz].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_final_flux_std, color="orange", linestyle="--", lw=3, label="Final flux (std osc)" )
                ax[0, i_cz].plot( E_values_GeV, np.power(E_values_GeV, energy_power)*flavor_final_flux_decoh, color="dodgerblue", linestyle=":", lw=3, label="Final flux (decoherence)" )

                # Plot ratios
                ax[1, i_cz].axhline( 1., color="grey", linestyle="-", lw=3)
                ax[1, i_cz].plot( E_values_GeV, ratio_std, color="orange", linestyle="--", lw=3)
                ax[1, i_cz].plot( E_values_GeV, ratio_decoh, color="dodgerblue", linestyle=":", lw=3)

                # Format
                for row in range(2) :
                    ax[row, i_cz].set_xscale("log")
                    ax[row, i_cz].set_xlim(E_values_GeV[0], E_values_GeV[-1])
                    ax[row, i_cz].grid(True)
                ax[0, i_cz].legend(fontsize=8)
                ax[0, i_cz].set_yscale("log")
                ax[0, i_cz].set_xticklabels([])
                ax[1, i_cz].set_xlabel(r"$E$ [GeV]")
                ax[0, i_cz].set_ylabel(  r"$" + (r"" if energy_power == 0. else r"E^{%0.3g}"%energy_power) + r"\phi" + r"$" )
                ax[1, i_cz].set_ylabel( "Final/initial flux" )

            # Format
            fig.tight_layout()


    # Save
    print("")
    dump_figures_to_pdf( __file__.replace(".py", "_"+matter+".pdf") )
