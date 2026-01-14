'''
Same as plot_delta_flux_tau_regeneration.py, but adding decoherence operator.

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

    E_GeV = 1e4 # Inject delta function flux at this energy
    E_values_GeV = np.geomspace(1., E_GeV, num=50)
    E_scale = "log"

    coszen = -1. # Plotting Earth-crossing neutrinos
    coszen_values = np.array([-1., 0., +1.])

    initial_states = [
        (0, False), # nue
        (0, True), # nuebar
        (1, False), # numu
        (1, True), # numubar
        (2, False), # nutau
        (2, True), # nutaubar
    ]

    # Configure decoherence model
    E0_eV = 1e9
    n = 0
    gamma_eV = get_vVBH_gamma_eV_from_coherence_length_km(EARTH_DIAMETER_km) # Coherence length = Earth diameter
    gamma0_eV = convert_gamma_eV_to_gamma_0_eV(gamma_eV=gamma_eV, E_eV=E_GeV*1e9, E0_eV=E0_eV, n=n)


    #
    # Create models
    #

    # Interactions but no tau decay
    calculator = OscCalculator(
        solver="nusquids",
        nusquids_variant="decoherence",
        atmospheric=True,
        energy_nodes_GeV=E_values_GeV,
        coszen_nodes=coszen_values,
        mixing_angles_rad=(0., 0., 0.), # Disable oscillations (make interaction effects clearer)
        matter="earth",
        interactions=True,
    )


    #
    # Loop over initial flux scenarios
    #

    # Loop over cases
    for initial_flavor, initial_nubar in initial_states :


        #
        # Generate and propagate flux
        #

        # Create delta function flux
        E_idx = (np.abs(E_values_GeV - E_GeV)).argmin()
        initial_flux = np.zeros( (E_values_GeV.size, coszen_values.size, calculator.num_neutrinos, 2) )
        initial_rho = int(initial_nubar)
        initial_flux[E_idx, :, initial_flavor, initial_rho] = 1.

        # Propagate (std osc)
        calculator.set_std_osc()
        final_flux = calculator.propagate_flux(
            initial_flux=initial_flux,
            energy_GeV=E_values_GeV,
            coszen=coszen_values,
        )

        # Propagate (decoherence)
        calculator.set_decoherence_model(model_name="randomize_state", gamma0_eV=gamma0_eV, n=n, E0_eV=E0_eV)
        final_flux_decoh = calculator.propagate_flux(
            initial_flux=initial_flux,
            energy_GeV=E_values_GeV,
            coszen=coszen_values,
        )


        #
        # Plot
        #

        # Make fig
        fig, ax = plt.subplots( ncols=2, nrows=calculator.num_neutrinos, figsize=(8, 2*calculator.num_neutrinos+1) )

        # Set a title
        initial_tex = calculator.get_nu_flavor_tex(initial_flavor, initial_nubar)
        fig.suptitle( r"$%s \rightarrow X$ // coszen = %0.3g" % (initial_tex, coszen) )

        # Loop over final states
        for final_flavor in range(calculator.num_neutrinos) :
            for final_nubar in [False, True] :
                final_rho = int(final_nubar)

                # Get ax for this flavor/nu(bar)
                this_ax = ax[final_flavor, final_rho]

                # Set a nice title
                final_tex = calculator.get_nu_flavor_tex(final_flavor, final_nubar)
                this_ax.set_title(r"$%s$"%final_tex)

                # Extract flux for this flavor
                coszen_idx = (np.abs(coszen_values - coszen)).argmin()
                flavor_initial_flux = initial_flux[:, coszen_idx, final_flavor, final_rho]
                flavor_final_flux = final_flux[:, coszen_idx, final_flavor, final_rho]
                flavor_final_flux_decoh = final_flux_decoh[:, coszen_idx, final_flavor, final_rho]

                # Handle tolerance (e.g. values very close to 0)
                tol = 1e-6
                tol_mask = flavor_final_flux < tol
                flavor_final_flux[tol_mask] = 0. 

                # Get sums
                initial_flux_sum = np.sum(flavor_initial_flux)
                final_flux_sum = np.sum(flavor_final_flux)
                delta_flux_sum = final_flux_sum - initial_flux_sum

                # Plot flux vs energy
                this_ax.step( E_values_GeV, flavor_initial_flux, color="black", linestyle="-", lw=2, label="Initial flux", where="mid" )
                this_ax.step( E_values_GeV, flavor_final_flux, color="orange", linestyle="--", lw=2, label="Final flux", where="mid" )
                this_ax.step( E_values_GeV, flavor_final_flux_decoh, color="dodgerblue", linestyle=":", lw=2, label="Final flux (decoherence)", where="mid" )

                # Format
                this_ax.set_xlabel(r"$E_\nu$ [GeV]")
                this_ax.set_xscale(E_scale)
                this_ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
                this_ax.legend(fontsize=8)
                this_ax.grid(True)
                this_ax.set_ylabel( "Flux per energy bin [arb. scale]", fontsize=8 )

            # Format
            fig.tight_layout()



    # Save
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
