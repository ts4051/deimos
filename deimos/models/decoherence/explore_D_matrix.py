'''
Exploring the various parameter permutations of the D matrix, including the underlying constraints

Tom Stuttard
'''

import sys, os, collections, copy, datetime

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *

from deimos.density_matrix_osc_solver.density_matrix_osc_solver import km_to_eV, GeV_to_eV
from deimos.utils.oscillations import calc_disappearance_prob_2flav_vacuum


#
# Helper functions/classes
#

class Envelope(object) :
    '''
    Track the envelope of many curves
    '''

    def __init__(self) :
        self.upper = None
        self.lower = None

    def update(self, values) :
        if self.upper is None :
            self.upper = values
            self.lower = values
        else :
            self.upper = np.maximum(values, self.upper)
            self.lower = np.minimum(values, self.lower)


    
def generate_random_D_matrix(calculator, random_state, disable_relaxation_params=False, disable_decoh_params=False, enforce_12_45_67_pairs=False) :
    '''
    Generate a random D matrix, and check if it is valid
    '''

    #TODO support beta terms

    #TODO doesn't really need to use calculator, need to move check_decoherence_D_matrix outside of the class

    # Generate a random diagonal D matrix
    D_diag = random_state.uniform(0., 1, size=9)
    D_diag[0] = 0.

    # Disable relaxation params, if requested
    if disable_relaxation_params :
        D_diag[3] = 0.
        D_diag[8] = 0.

    # Disable decoherence params, if requested
    if disable_decoh_params :
        D_diag[1] = 0.
        D_diag[2] = 0.
        D_diag[4] = 0.
        D_diag[5] = 0.
        D_diag[6] = 0.
        D_diag[7] = 0.

    # Enforce the 12/45/67 pair equalities, if requested
    if enforce_12_45_67_pairs :
        D_diag[2] = D_diag[1]
        D_diag[5] = D_diag[4]
        D_diag[7] = D_diag[6]

    # Form the full matrix
    D = np.diag(D_diag.tolist())

    # Test which cases are valid
    try :
        calculator.check_decoherence_D_matrix(D)
        valid = True
    except Exception as e:
        # print(str(e))
        valid = False

    return D, valid


#
# Plot functions
#

def compare_D_matrix_textures(solver, num_points=100) :
    '''
    Compare decoherence resulting from different textures of the D matrix

    (a) Diagonal
    (b) Off-diagonal

    Respecting all constraints between parameters
    '''


    #
    # Create model
    # 

    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
    )

    # Vacuum
    calculator.set_matter("vacuum")

    # Define plot range
    initial_flavor, final_flavor = 1, 1
    nubar = False
    E_GeV_values = np.logspace(2., 5., num=num_points)
    coszen_values = np.linspace(-1, 0., num=num_points)
    coszen_ref = -1.


    #
    # Loop over n
    #

    E0_eV = 1e12 # 1 TeV

    n_cases = [
        {
            "n" : 0,
            "E_GeV_ref" : 1e3,
            "gamma0_eV" : 1.18e-15 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
        {
            "n" : 2,
            "E_GeV_ref" : 1e5,
            "gamma0_eV" : 9.80e-18 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
    ]

    for n_case in n_cases :

        print("n = %i" % n_case["n"])


        #
        # Non-relaxation term (phase perturbation cases)
        #

        # Define cases
        gamma_cases = [
            {
                "D_matrix_eV" : None,
                "color" : "grey",
                "linestyle" : "-",
                "label" : "Standard osc",
            },
            {
                "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
                # "gamma_scaling" : 1,
                "color" : "black",
                "linestyle" : "-",
                # "label" : r"$\Gamma_1 = \Gamma_2 = \Gamma_4 = \Gamma_5 = \Gamma_6 = \Gamma_7 = \Gamma$",
                "label" : r"Phase perturbation (PP)",
            },
            {
                "D_matrix_eV" : np.diag([0., 0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
                # "gamma_scaling" : 1.1,
                "color" : "magenta",
                "linestyle" : "--",
                "label" : r"PP w/ $\Gamma_1 = \Gamma_2 = 0$",
            },
            {
                "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
                # "gamma_scaling" : 1.1,
                "color" : "orange",
                "linestyle" : "--",
                "label" : r"PP w/ $\Gamma_4 = \Gamma_5 = 0$",
            },
            {
                "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0.]),
                # "gamma_scaling" : 4.,
                "color" : "dodgerblue",
                "linestyle" : "--",
                "label" : r"PP w/ $\Gamma_6 = \Gamma_7 = 0$",
            },
            # { # Forbidden
            #     "D_matrix_eV" : np.diag([0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0., 0., 0., 0.]),
            #     "color" : "magenta",
            #     "linestyle" : "--",
            #     "label" : r"$\Gamma_1 = \Gamma_2 = \Gamma$, rest = 0",
            # },
            # {
            #     "D_matrix_eV" : np.diag([0.,0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0., 0., 0.]),
            #     "color" : "magenta",
            #     "linestyle" : "--",
            #     "label" : r"$\Gamma_4 = \Gamma_5 = \Gamma$, rest = 0",
            # },
            # {
            #     "D_matrix_eV" : np.diag([0., 0., 0., 0., 0., 0., n_case["gamma0_eV"], n_case["gamma0_eV"], 0.]),
            #     "color" : "magenta",
            #     "linestyle" : "--",
            #     "label" : r"$\Gamma_6 = \Gamma_7 = \Gamma$, rest = 0",
            # },
        ]

        # Make figure
        fig_E, ax_E = None, None
        fig_cz, ax_cz = None, None

        # Loop over cases
        for gamma_case in gamma_cases :

            print(gamma_case["label"])

            # Set model
            if gamma_case["D_matrix_eV"] is None :
                calculator.set_std_osc()
            else :
                calculator.set_decoherence_D_matrix(D_matrix_eV=gamma_case["D_matrix_eV"], n=n_case["n"], E0_eV=E0_eV)

            # Plot vs coszen
            fig_cz, ax_cz, _ = calculator.plot_osc_prob_vs_distance( 
                initial_flavor=initial_flavor,
                final_flavor=final_flavor,
                energy_GeV=n_case["E_GeV_ref"], 
                coszen=coszen_values, 
                nubar=nubar, 
                fig=fig_cz, 
                ax=ax_cz, 
                label=gamma_case["label"], 
                color=gamma_case["color"], 
                linestyle=gamma_case["linestyle"], 
                xscale="linear",
                ylim=(-0.05, 1.05),
                lw=2,
            )

            # Plot vs E
            fig_E, ax_E, _ = calculator.plot_osc_prob_vs_energy( 
                initial_flavor=initial_flavor,
                final_flavor=final_flavor,
                energy_GeV=E_GeV_values, 
                coszen=coszen_ref, 
                nubar=nubar, 
                fig=fig_E, 
                ax=ax_E, 
                label=gamma_case["label"], 
                color=gamma_case["color"], 
                linestyle=gamma_case["linestyle"], 
                xscale="log",
                ylim=(-0.05, 1.05),
                lw=2,
            )

        # Titles
        fig_E.suptitle(r"$n$ = %i, $\Gamma_0$ = %0.3g eV, $\cos(\theta)$ = %0.3g" % (n_case["n"], n_case["gamma0_eV"], coszen_ref) )
        fig_cz.suptitle(r"$n$ = %i, $\Gamma_0$ = %0.3g eV, $E_\nu$ = %0.3g GeV" % (n_case["n"], n_case["gamma0_eV"], n_case["E_GeV_ref"]) )

        # Format
        for ax in [ax_E, ax_cz] :
            ax[0].legend(fontsize=10)
        for fig in [fig_E, fig_cz] :
            fig.tight_layout()



        #
        # Including relaxation terms (state selection cases)
        #

        #TODO


def compare_random_D_matrix_textures(solver, num_models=100) :
    '''
    Generate random (valid) D matrix textures and compare them
    '''

    start_time = datetime.datetime.now()

    # Init RNG
    random_state = np.random.RandomState(12345)


    #
    # Create model
    # 

    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
    )

    # Vacuum
    calculator.set_matter("vacuum")

    # Define plot range
    initial_flavor, final_flavor = 1, 1
    nubar = False
    E_GeV_values = np.geomspace(500, 100e3, num=num_points)
    coszen_values = np.linspace(-1, 0., num=num_points)
    coszen_ref = -1.


    #
    # Loop over n
    #

    E0_eV = 1e12 # 1 TeV

    n_cases = [
        {
            "n" : 0,
            "E_GeV_ref" : 1e3,
            "gamma0_eV" : 1.18e-15 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
        {
            "n" : 2,
            "E_GeV_ref" : 1e5,
            "gamma0_eV" : 9.80e-18 * 5., # https://arxiv.org/pdf/2308.00105.pdf table 1 (90% CL), multiplied by 5 for larger signal
        },
    ]

    for n_case in n_cases :

        print("n = %i" % n_case["n"])


        #
        # Plot our cases
        #

        # Choose whether to plot individual models, or the envelope
        plot_individual_models = False
        first_model_plot = True
        osc_probs_envelope_cz, osc_probs_envelope_E = Envelope(), Envelope()

        # Make figures
        fig, ax = plt.subplots(ncols=2, figsize=(10,4))

        # Plot helper function
        def _plot(color, linestyle, lw, zorder, label=None, alpha=1., update_envelope=False) :
            # Common args
            plot_kw = dict(
                initial_flavor=initial_flavor,
                final_flavor=final_flavor,
                nubar=nubar, 
                ylim=(-0.05, 1.05),
                label=label,
                color=color,
                linestyle=linestyle,
                lw=lw,
                zorder=zorder,
                alpha=alpha,
            )
            # Plot vs coszen
            _, _, osc_probs_cz = calculator.plot_osc_prob_vs_distance( 
                energy_GeV=n_case["E_GeV_ref"], 
                coszen=coszen_values, 
                fig=fig, 
                ax=[ax[0]], 
                xscale="linear",
                **plot_kw,
            )
            osc_probs_cz = osc_probs_cz[...,final_flavor]
            # Plot vs E
            _, _, osc_probs_E = calculator.plot_osc_prob_vs_energy( 
                energy_GeV=E_GeV_values, 
                coszen=coszen_ref, 
                fig=fig, 
                ax=[ax[1]], 
                xscale="log",
                **plot_kw,
            )
            osc_probs_E = osc_probs_E[...,final_flavor]
            if update_envelope :
                osc_probs_envelope_cz.update(osc_probs_cz)
                osc_probs_envelope_E.update(osc_probs_E)
            return osc_probs_cz, osc_probs_E

        # Plot standard osc
        calculator.set_std_osc()
        _plot(color="black", linestyle="-", lw=3, label="Standard osc", zorder=5, update_envelope=False)

        # Plot phase perturbation model
        calculator.set_decoherence_model("randomize_phase", gamma0_eV=n_case["gamma0_eV"], n=n_case["n"], E0_eV=E0_eV)
        phase_perturbation_color = "red"
        _plot(color=phase_perturbation_color, linestyle="-", lw=3, label="Phase perturbation", zorder=6, update_envelope=True)

        # Plot state selection
        calculator.set_decoherence_model("randomize_state", gamma0_eV=n_case["gamma0_eV"], n=n_case["n"], E0_eV=E0_eV)
        state_selection_color = "blue"
        _plot(color=state_selection_color, linestyle="-", lw=3, label="State selection", zorder=6, update_envelope=True)

        # Add some other cases, in particular those known to be bounding
        if False :
            calculator.set_decoherence_D_matrix(D_matrix_eV=np.diag([0., 0., 0., 0., 1., 1., 1., 1., 0.])*n_case["gamma0_eV"], n=n_case["n"], E0_eV=E0_eV)
            _plot(color="orange", linestyle=":", lw=3, label=r"$\Gamma_3 = \Gamma_1 = \Gamma_2 = \Gamma_8 = 0$", zorder=5, update_envelope=True)

            calculator.set_decoherence_D_matrix(D_matrix_eV=np.diag([0., 1., 1., 0., 0., 0., 1., 1., 0.])*n_case["gamma0_eV"], n=n_case["n"], E0_eV=E0_eV)
            _plot(color="orange", linestyle="-.", lw=3, label=r"$\Gamma_3 = \Gamma_4 = \Gamma_5 = \Gamma_8 = 0$", zorder=5, update_envelope=True)

        calculator.set_decoherence_D_matrix(D_matrix_eV=np.diag([0., 1., 1., 0., 1., 1., 0., 0., 0.])*n_case["gamma0_eV"], n=n_case["n"], E0_eV=E0_eV) # This appears to be the lower bound for numu survival
        _plot(color="orange", linestyle="-", lw=3, label=r"$\Gamma_3 = \Gamma_6 = \Gamma_7 = \Gamma_8 = 0$", zorder=5, update_envelope=True)


        #
        # Generate random models and plot resulting transition probabilities
        #

        #TODO Better to analytically find bounding cases if possible, but difficult as system gains DOFs

        # Handle both with and without relaxation params cases           #TODO beta
        # for (disable_relaxation_params, enforce_12_45_67_pairs, color) in [ (True, True, phase_perturbation_color), (False, False, state_selection_color) ] :
        for (disable_relaxation_params, enforce_12_45_67_pairs, color) in [ (False, False, state_selection_color), (True, True, phase_perturbation_color) ] :

            # Loop to generate models
            trial_counter, model_counter = 0, 0
            ymin_E, ymax_E, ymin_cz, ymax_cz = None, None, None, None
            num_models_to_plot = min(num_models, 30)
            while model_counter < num_models :

                # Report
                print("Trial %i (valid models so far = %i of %i)" % (trial_counter, model_counter, num_models))
                trial_counter += 1

                # Generate random D matrix
                D, valid = generate_random_D_matrix(
                    calculator=calculator,
                    random_state=random_state, 
                    disable_relaxation_params=disable_relaxation_params, 
                    disable_decoh_params=False, 
                    enforce_12_45_67_pairs=enforce_12_45_67_pairs,
                )

                # Only proceed if valid
                if valid :

                    # Book-keeping
                    # model_color = cmap(float(model_counter)/float(num_models))
                    # model_color = "seagreen"
                    model_counter += 1

                    # Scale such that the largest D matrix element = gamma0
                    D = D * n_case["gamma0_eV"] * (1. / np.max(D))    #TODO how exactly to do this? Is this coherence length of largest element? think so

                    # Set model
                    calculator.set_decoherence_D_matrix(D_matrix_eV=D, n=n_case["n"], E0_eV=E0_eV)

                    # Get osc probs for this model
                    osc_probs_E = calculator.calc_osc_prob(
                        initial_flavor=initial_flavor,
                        nubar=nubar, 
                        energy_GeV=E_GeV_values,
                        coszen=coszen_ref,
                    )
                    osc_probs_E = osc_probs_E[:,0,final_flavor]
                    assert osc_probs_E.ndim == 1

                    osc_probs_cz = calculator.calc_osc_prob(
                        initial_flavor=initial_flavor,
                        nubar=nubar, 
                        energy_GeV=n_case["E_GeV_ref"],
                        coszen=coszen_values,
                    )
                    osc_probs_cz = osc_probs_cz[0,:,final_flavor]
                    assert osc_probs_cz.ndim == 1

                    # Update envelopes
                    osc_probs_envelope_cz.update(osc_probs_cz)
                    osc_probs_envelope_E.update(osc_probs_E)

                    # Plot some examples
                    if plot_individual_models :
                        if model_counter < num_models_to_plot :
                            for (i, x, y) in zip([0, 1], [coszen_values, E_GeV_values], [osc_probs_cz, osc_probs_E]) :
                                ax[i].plot( x, y, color="black", alpha=0.1, lw=1, zorder=3, label=("Random models" if first_model_plot else None))
                            first_model_plot = False

        # Plot envelope
        for (i, x, y) in zip([0, 1], [coszen_values, E_GeV_values], [osc_probs_envelope_cz, osc_probs_envelope_E]) :
            ax[i].fill_between( x, y.lower, y.upper, color="magenta", alpha=0.2, zorder=3, label="All models (envelope)")

        # Titles
        ax[0].set_title(r"$\cos(\theta)$ = %0.3g" % (coszen_ref) )
        ax[1].set_title(r"$E_\nu$ = %0.3g GeV" % (n_case["E_GeV_ref"]) )
        fig.suptitle(r"$n$ = %i, $\Gamma_0$ = %0.3g eV" % (n_case["n"], n_case["gamma0_eV"]) )

        # Format
        for this_ax in ax :
            this_ax.legend(fontsize=8, loc="lower right", ncol=2)
        fig.tight_layout()

        # Save
        for ext in ["png", "pdf"] :
            fig.savefig("compare_random_D_matrix_textures_n%i.%s"%(n_case["n"], ext))

    # Report time (can be slow)
    time_taken = datetime.datetime.now() - start_time
    print("Random D matrix comparisons complete : %i models per case : Took %s" % (num_models, time_taken))




def explore_D_matrix_constraints(solver, num_points=100) :
    '''
    Explore what D matrix element values are actually possible, given the various constraints
    '''


    #
    # Create model
    # 

    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
    )

    calculator.set_matter("vacuum")


    #
    # Test all gammas, and project
    #

    '''
    Randomly generate D matrices and see what works, to test physicality constraints

    Things I have learned by doing this (all assuming diagonal only so far):

      (a) If relaxation params are 0 :
         - g1 = g2, g4 = g5, g6 = g7 pair conditions must be respected    (a1-7 conditions imply g1==g2, g=<=g5, g6==g7)
         - there is no relation between the pairs, e.g. g1,4,6, they can take any value w.r.t. each other
         - If one g pair = 0 , the other two pairs MUST equal each other. this is a conical bound I think (consequence of a7)
      (b) If relaxation params are >0 :
         - g1 = g2, g4 = g5, g6 = g7 pair conditions not longer need to be respected    (a1-7 conditions have more freedom due to non-zero g3/8)
         - There are relations between g1/2/4/5/6/7 and g3/8, but still plenty of freedom
         - There is a clear relation limit relating g3 and g8, which does not allow small g3 and large g8 simultaneously  (a1,2,3 conditions combined => g8 >= 3 * g3)
         - There is a clear relation limit between g1/2 and g8, but not for the g4/5/6/7   => g8 >= 3 * g1/2   (ceomes from a1,2,3 conditions, which only affect g1/2/3/8. g3 not limited do to lack of factor 3 there)
    '''

    N = 10000
    random_state = np.random.RandomState(12345)

    disable_relaxation_params = True
    disable_decoh_params = True
    enforce_12_45_67_pairs = True

    valid_D_matrices, invalid_D_matrices = [], []
    for n in range(N) :

        # Generate random D matrix
        D, valid = generate_random_D_matrix(
            calculator=calculator,
            random_state=random_state, 
            disable_relaxation_params=disable_relaxation_params, 
            disable_decoh_params=disable_decoh_params, 
            enforce_12_45_67_pairs=enforce_12_45_67_pairs,
        )

        # Store
        if valid :
            valid_D_matrices.append( np.diagonal(D).tolist() )
        else :
            invalid_D_matrices.append( np.diagonal(D).tolist() )

    valid_D_matrices = np.array(valid_D_matrices)
    invalid_D_matrices = np.array(invalid_D_matrices)

    assert len(valid_D_matrices) > 0, "No valid D matrices found"

    print("Found %i valid D matrices, %i invalid" % (len(valid_D_matrices), len(invalid_D_matrices)))

    # Plot pairs...
    # Loop over pairs of interest
    for (i, j) in [ (1, 2), (4, 5), (6, 7), (1, 4), (1, 6), (4, 6), (3, 8), (1, 3), (1, 8), (4, 3), (4, 8), (6, 3), (6, 8) ] :

        # Scatter plots param pairs
        fig, ax = plt.subplots(figsize=(6,5))
        # if np.shape(invalid_D_matrices)[0] > 0 :
        #     ax.scatter(invalid_D_matrices[:,i], invalid_D_matrices[:,j], color="red", marker=".")
        if np.shape(valid_D_matrices)[0] > 0 :
            ax.scatter(valid_D_matrices[:,i], valid_D_matrices[:,j], color="green", marker=".")

        # Add conditions of interest
        if (i in [1,2,3]) and (j==8) :
            x = np.linspace(0., 1., num=num_points)
            y = 3.*x
            ax.plot(x, y, color="black", linestyle="--", label=r"$\Gamma_{%i} = 3 \Gamma_{%i}$"%(j,i))
        # if (i in [1,2]) and (j==3) :
        #     x = np.linspace(0., 1., num=num_points)
        #     y = 3.*x/2.
        #     ax.plot(x, y, color="black", linestyle="--", label=r"$\Gamma_{%i} = 3 \Gamma_{%i} / 2$"%(j,i))

        # Format
        ax.set_xlabel(r"$\Gamma_{%i}$"%i)
        ax.set_ylabel(r"$\Gamma_{%i}$"%j)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()





#
# Main
#

if __name__ == "__main__" :

    # Get args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Num scan point")
    args = parser.parse_args()

    # Run plotting functions    
    explore_D_matrix_constraints(solver=args.solver, num_points=args.num_points)
    # compare_random_D_matrix_textures(solver=args.solver, num_models=10)
    compare_D_matrix_textures(solver=args.solver, num_points=args.num_points)

    # Done
    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
