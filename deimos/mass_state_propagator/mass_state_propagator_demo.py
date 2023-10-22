'''
Example of using the mass state propagator

Tom Stuttard
'''

from deimos.density_matrix_osc_solver.density_matrix_osc_solver import get_pmns_matrix, psi_flav_to_mass_basis, psi_mass_to_flav_basis, psi_flavor_prob
from deimos.mass_state_propagator.mass_state_propagator import MassStatePropagator

from deimos.utils.plotting import *


#
# Globals: Physics steering
#

# These plots are 2 flavor for simplicity
NUM_STATES = 2

# Use same neutrino params across the board
# Choosing values that are not crazy but not realistic either, but chosen for plot clarity
INITIAL_FLAVOR = 0
FINAL_FLAVOR = 0
E_GeV = 1.
LOWEST_NEUTRINO_MASS_eV = 1.e-1
MASSES_eV = np.array([ LOWEST_NEUTRINO_MASS_eV, np.sqrt(2.)*LOWEST_NEUTRINO_MASS_eV ]) # 2x wavelength
DELTA_M2_eV2 = np.square(MASSES_eV[1]) - np.square(MASSES_eV[0])
print( "m1 = %0.3g eV : m2 = %0.3g eV : dm2 = %0.3g eV^2" % (MASSES_eV[0], MASSES_eV[1], DELTA_M2_eV2) )
BASELINE_SHORT_km = 250. # 1 osc wavelength 
BASELINE_LONG_km = 1250. # 5x longer to show damping effect
PERTURBATION_MEAN_FREE_PATH_km = BASELINE_SHORT_km # Same as osc wavelength
OFF_MAXIMAL_THETA_rad = np.deg2rad(20.)


#
# Globals: Plot steering
#

FLAVOR_TEX = [ r"\alpha", r"\beta"]
INITIAL_TEX = FLAVOR_TEX[INITIAL_FLAVOR]
FINAL_TEX = FLAVOR_TEX[INITIAL_FLAVOR]
TRANSITION_PROB_TEX = r"P($\nu_%s \rightarrow \nu_%s$)" % (INITIAL_TEX, FINAL_TEX)
DISTANCE_TEX = r"Propagation distance [km]"


#
# Plotting functions
#

def demonstrate_model(num_events=1000) :

    #
    # Define system
    #

    # Maximal mixing
    theta = np.array([np.deg2rad(45.)])
    PMNS = get_pmns_matrix(theta)

    perturbation = "randomize_phase"


    #
    # Create model
    #

    model = MassStatePropagator(
        num_states=NUM_STATES,
        mass_state_masses_eV=MASSES_eV,
        PMNS=PMNS,
        seed=12345,
    )


    #
    # Plot standard case (no perturbations)
    #

    x_obs, x_nu, psi_mass_values_std_osc, psi_flav_values_std_osc, osc_prob_values_std_osc = model.get_osc_probs(
        initial_flavor=INITIAL_FLAVOR,
        E_GeV=E_GeV,
        distance_km=BASELINE_SHORT_km,
        perturbation=None,
    )

    figsize = (6,6)
    fig, ax = plt.subplots( nrows=3, sharex=True, figsize=figsize )

    show_imaginary = True
    imag_alpha = 0.3

    # Plot flavor basis
    ax[0].plot( x_obs, psi_flav_values_std_osc[:,0].real, color="red", linestyle="-", label=r"$\nu_{\alpha}$ real" )
    if show_imaginary :
        ax[0].plot( x_obs, psi_flav_values_std_osc[:,0].imag, color="red", linestyle="--", label=r"$\nu_{\alpha}$ imaginary", alpha=imag_alpha )
    ax[0].plot( x_obs, psi_flav_values_std_osc[:,1].real, color="blue", linestyle="-", label=r"$\nu_{\beta}$ real" )
    if show_imaginary :
        ax[0].plot( x_obs, psi_flav_values_std_osc[:,1].imag, color="blue", linestyle="--", label=r"$\nu_{\beta}$ imaginary", alpha=imag_alpha )
    ax[0].set_ylim(-1.1,1.1)
    ax[0].set_ylabel(r"$\left|\nu_{\rm{flavor}}\right\rangle$")

    # Plot mass basis
    ax[1].plot( x_obs, psi_mass_values_std_osc[:,0].real, color="orange", linestyle="-", label=r"$\nu_{1}$ real" )
    if show_imaginary : 
        ax[1].plot( x_obs, psi_mass_values_std_osc[:,0].imag, color="orange", linestyle="--", label=r"$\nu_{1}$ imaginary", alpha=imag_alpha )
    ax[1].plot( x_obs, psi_mass_values_std_osc[:,1].real, color="purple",linestyle="-", label=r"$\nu_{2}$ real" )
    if show_imaginary : 
        ax[1].plot( x_obs, psi_mass_values_std_osc[:,1].imag, color="purple",linestyle="--", label=r"$\nu_{2}$ imaginary", alpha=imag_alpha )
    ax[1].set_ylim(-1.1,1.1)
    ax[1].set_ylabel(r"$\left|\nu_{\rm{mass}}\right\rangle$")

    # Plot osc prob
    ax[2].plot( x_obs, osc_prob_values_std_osc[:,0], color="red", label=r"P($\nu_{\alpha} \rightarrow \nu_{\alpha}$)" )
    ax[2].plot( x_obs, osc_prob_values_std_osc[:,1], color="blue", label=r"P($\nu_{\alpha} \rightarrow \nu_{\beta}$)" )
    ax[2].set_ylim(-0.05,1.05)
    ax[2].set_ylabel(r"$P(X \rightarrow Y)$")

    # Format
    ax[2].set_xlabel(DISTANCE_TEX)
    # ax[2].set_xticklabels([])
    for this_ax in ax :
        width = x_obs[-1] - x_obs[0]
        padding = width / 100.
        this_ax.set_xlim(x_obs[0]-padding, x_obs[-1]+padding)
        this_ax.grid(True,zorder=-1)
        # this_ax.legend()
        this_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    fig.tight_layout()


    #
    # Plot with perturbations
    #

    # Generate some perturbed neutrinos
    _, x_nu, psi_mass_per_event_perturb, psi_flav_per_event_perturb, osc_prob_per_event_perturb, osc_prob_mean_perturb = model.get_osc_probs_loop(
        num_events=100,
        initial_flavor=INITIAL_FLAVOR,
        E_GeV=E_GeV,
        distance_km=BASELINE_SHORT_km,
        perturbation=perturbation,
        num_individual_events_to_return=500,
        perturbation_options = { "mean_free_path_km" : PERTURBATION_MEAN_FREE_PATH_km },
    )

    fig, ax = plt.subplots( nrows=3, sharex=True, figsize=figsize )

    show_imaginary = False
    imag_alpha = 0.3
    std_linestyle = ":"
    decoh_linestyle = "-"

    i_event = 43 # 28 35 43

    # Plot flavor basis
    ax[0].plot( x_obs, psi_flav_values_std_osc[:,0].real, color="red", linestyle=std_linestyle, label=r"$\nu_{\alpha}$ unperturbed" )
    ax[0].plot( x_obs, psi_flav_per_event_perturb[i_event,:,0].real, color="red", linestyle=decoh_linestyle, label=r"$\nu_{\alpha}$ perturbed" )
    ax[0].plot( x_obs, psi_flav_values_std_osc[:,1].real, color="blue", linestyle=std_linestyle, label=r"$\nu_{\beta}$ unperturbed" )
    ax[0].plot( x_obs, psi_flav_per_event_perturb[i_event,:,1].real, color="blue", linestyle=decoh_linestyle, label=r"$\nu_{\beta}$ perturbed" )
    ax[0].set_ylim(-1.1,1.1)
    ax[0].set_ylabel(r"$\left|\nu_{\rm{flavor}}\right\rangle$ (real)")

    # Plot mass basis
    ax[1].plot( x_obs, psi_mass_values_std_osc[:,0].real, color="orange", linestyle=std_linestyle, label=r"$\nu_{1}$ unperturbed" )
    ax[1].plot( x_obs, psi_mass_per_event_perturb[i_event,:,0].real, color="orange", linestyle=decoh_linestyle, label=r"$\nu_{1}$ perturbed" )
    ax[1].plot( x_obs, psi_mass_values_std_osc[:,1].real, color="purple", linestyle=std_linestyle, label=r"$\nu_{2}$ unperturbed" )
    ax[1].plot( x_obs, psi_mass_per_event_perturb[i_event,:,1].real, color="purple", linestyle=decoh_linestyle, label=r"$\nu_{2}$ perturbed" )
    ax[1].set_ylim(-1.1,1.1)
    ax[1].set_ylabel(r"$\left|\nu_{\rm{mass}}\right\rangle$ (real)")

    # Plot osc prob
    ax[2].plot( x_obs, osc_prob_values_std_osc[:,0], color="red", linestyle=std_linestyle, label=r"P($\nu_{\alpha} \rightarrow \nu_{\alpha}$) unperturbed" )
    ax[2].plot( x_obs, osc_prob_per_event_perturb[i_event,:,0], color="red", linestyle=decoh_linestyle, label=r"P($\nu_{\alpha} \rightarrow \nu_{\alpha}$) perturbed" )
    ax[2].plot( x_obs, osc_prob_values_std_osc[:,1], color="blue", linestyle=std_linestyle, label=r"P($\nu_{\alpha} \rightarrow \nu_{\beta}$) unperturbed" )
    ax[2].plot( x_obs, osc_prob_per_event_perturb[i_event,:,1], color="blue", linestyle=decoh_linestyle, label=r"P($\nu_{\alpha} \rightarrow \nu_{\beta}$) perturbed" )
    ax[2].set_ylim(-0.05,1.05)
    ax[2].set_ylabel(r"$P(X \rightarrow Y)$")

    # Format
    ax[2].set_xlabel(DISTANCE_TEX)
    # ax[2].set_xticklabels([])
    for this_ax in ax :
        width = x_obs[-1] - x_obs[0]
        padding = width / 100.
        this_ax.set_xlim(x_obs[0]-padding, x_obs[-1]+padding)
        this_ax.grid(True,zorder=-1)
        # this_ax.legend()
        this_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    fig.tight_layout()


    #
    # Show net impact on oscillation probability
    #

    # Re-calculate unoperturbed case at the longer baseline
    x_obs, x_nu, psi_mass_values_std_osc, psi_flav_values_std_osc, osc_prob_values_std_osc = model.get_osc_probs(
        initial_flavor=INITIAL_FLAVOR,
        E_GeV=E_GeV,
        distance_km=BASELINE_LONG_km,
        perturbation=None,
    )

    # Generate a lot more perturbed neutrinos, and over a longer distance to see damping
    _, x_nu, psi_mass_per_event_perturb, psi_flav_per_event_perturb, osc_prob_per_event_perturb, osc_prob_mean_perturb = model.get_osc_probs_loop(
        num_events=num_events,
        initial_flavor=INITIAL_FLAVOR,
        E_GeV=E_GeV,
        distance_km=BASELINE_LONG_km,
        perturbation=perturbation,
        num_individual_events_to_return=500,
        perturbation_options = { "mean_free_path_km" : PERTURBATION_MEAN_FREE_PATH_km },
    )

    fig, ax = plt.subplots( figsize=(6,4) )

    # Plot the standard osc picture
    ax.plot( x_obs, osc_prob_values_std_osc[:,FINAL_FLAVOR], color="blue", linestyle="-", zorder=2, label=r"$\nu_{\rm{unperturbed}}$" )

    # Plot the individual perturbed curves
    max_events_to_plot = 75
    for i_event in range( min( max_events_to_plot, osc_prob_per_event_perturb.shape[0] ) ) :
        ax.plot( x_obs, osc_prob_per_event_perturb[i_event,:,FINAL_FLAVOR], color="red", alpha=0.05, zorder=1, label=None )

    # Make a clearer legend element for the individual nu (alpha is too low otherwise)
    xlim = ax.get_xlim()
    ax.plot( [-101,-100], [0.,0.], color="red", alpha=0.4, label=r"$\nu_{\rm{perturbed}}$" )
    ax.set_xlim(xlim) 

    # Plot the average of all perturbed curves
    ax.plot( x_obs, osc_prob_mean_perturb[:,FINAL_FLAVOR], color="red", linestyle="--", zorder=3, label=r"$\nu_{\rm{perturbed,average}}$" )

    # Format
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0., BASELINE_LONG_km)
    ax.set_xlabel(DISTANCE_TEX)
    # ax.set_xticklabels([])
    ax.set_ylabel( r"P($\nu_%s \rightarrow \nu_%s$)" % (INITIAL_TEX,FINAL_TEX) )
    ax.grid(True)
    ax.legend( loc="upper right")
    fig.tight_layout()



if __name__ == "__main__" :

    # Make plots
    demonstrate_model(num_events=100)

    # Dump figures to PDF
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )

