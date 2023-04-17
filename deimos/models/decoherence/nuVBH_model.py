'''
Mathematical definition of my neutrino - virtual black hole interaction model
and the resulting decoherence operators

See https://arxiv.org/abs/2007.00068 for details.

Tom Stuttard
'''

from utils.plotting.standard_modules import *

from analysis.common.utils.natural_units import si_to_natural_units, natural_to_si_units
from analysis.oscNext_decoherence.theory.constants import PLANCK_LENGTH_m, PLANCK_MASS_eV

from pisa import ureg


#
# Globals
#

# Choose colors to represent my models
MODEL_COLORS = {
    "randomize_phase" : "red",
    "randomize_mass_state" : "blue",
    "randomize_state" : "blue",
    "randomize_flavor" : "magenta",
    "neutrino_loss" : "green",
}

# Labels for the models
MODEL_LABELS = collections.OrderedDict()
MODEL_LABELS["randomize_phase"] = r"$\nu$-VBH: Phase perturbation"
MODEL_LABELS["randomize_state"] = r"$\nu$-VBH: Mass/flavor selected"
MODEL_LABELS["neutrino_loss"] = r"$\nu$-VBH: $\nu$ loss"

# Most papers defined the energy scale at 1 GeV
DEFAULT_GAMMA_ENERGY_SCALE_eV = 1.e9


#
# gamma enegy dependence
#

def get_gamma(gamma0_eV, E_eV, E0_eV, n) : #TODO Integrate into DMOS?
    '''
    Compute gamma(E)
    '''
    return gamma0_eV * ( (E_eV / E0_eV) ** n )


def convert_gamma_eV_to_gamma_0_eV(gamma_eV, E_eV, E0_eV, n) :
    '''
    Compute the value of gamma and E0 (e.g. gamma0) from the desired/observed damping strength gamma
    '''
    return gamma_eV * ( ( E0_eV / E_eV ) ** n )


def convert_gamma0_energy_scale(gamma0_in_eV, energy_scale_in_eV, energy_scale_out_eV) :
    '''
    Convert from one energy scale to another for the new decoherence gamma_0 matrix definition.
    A common use case is converting from the 1 GeV scale many papers use to the Planck scale.
    NOTE: This is NOT converting to the dimensonless parameter used to comapre to the Planck scale in e.g. hep-ph/0506168v1.pdf

    Converting according to: ( gamma_0,A / Lambda_0,A ) = ( gamma_0,B / Lambda_0,B ) , where Lamdba is an energy scale.
    '''
    return energy_scale_out_eV * ( gamma0_in_eV / energy_scale_in_eV )


#TODO interdepedence equation


#
# Conversion to distance scale
#

def convert_gamma_eV_to_gamma_inv_m(gamma_eV) :
    '''
    invert gamma [eV] to get 1/gamma (e.g. distance scale) in [m]
    '''
    return natural_to_si_units( 1./(gamma_eV*ureg["eV"]), "length" ).m_as("m")


def convert_gamma_inv_m_to_gamma_eV(gamma_inv_m):
    '''
    Re-invert 1/gamma [m] to get gamma [eV]
    '''
    return ( 1. / si_to_natural_units( gamma_inv_m*ureg["m"] ).m_as("1/eV") )


#
# Planck scale gamma
#

def get_gamma_planck(zeta_p, E_eV, n) :
    '''
    Comute gamma according to the Planck scale definition from hep-ph/0506168
    '''
    return zeta_p * (E_eV**n) / (PLANCK_MASS_eV**(n-1.))


def convert_gamma_eV_to_zeta_planck(gamma_eV, E_eV, n) :
    '''
    Compute the value of lambda from the desired/observed damping strength gamma
    '''
    return gamma_eV * (PLANCK_MASS_eV**(n-1.))/ (E_eV**n)


def convert_gamma0_to_zeta_planck(gamma0_eV, E0_eV, n) :
    '''
    Get the dimensionless constant of proportionality relative to the natural Planck scale
    '''
    return gamma0_eV * ( PLANCK_MASS_eV ** (n-1.) ) / ( E0_eV ** n )


def convert_zeta_planck_to_gamma0(zeta_p, E0_eV, n) :
    '''
    inverse of `convert_lambda_Planck_to_gamma0`
    '''
    return zeta_p * ( E0_eV ** n ) / ( PLANCK_MASS_eV ** (n-1.) )


def gamma_inv_planck_m__method2( zeta_p, E_eV, n) :
    '''
    This is intended only as a cross-check method of calculating the Planck scale decoherence length.
    1) Express gamma as gamma = Mp * (E/Mp)^n so that only have the units of the Planck mass
    2) Re-express Mp in units of [1/m] byt dividing through by hbar*c expressed in units of [eV.m] #TODO Is this just the Planck length?
    Note: This is the same as: gamma = ( (Planck length) / lambda ) * (Mp/E)^n
    '''
    gamma_eV = zeta_planck * PLANCK_MASS_eV * ((E_eV/PLANCK_MASS_eV)**n)
    gamma_per_m = gamma_eV / ( 197e-9 ) # This is hbar*c in [eV*m] 
    gamma_inv_m = 1. / gamma_per_m
    return gamma_inv_m


def get_osc_prob_at_coherence_length(long_distance_prob_limit, initial_flavor, final_flavor) :
    '''
    Coherence definition is distance at which damping term -> 1/e
    '''

    damping_amplitude = np.exp(-1.)

    if initial_flavor == final_flavor :
        # Damping from 1 -> limit
        return ( (1. - long_distance_prob_limit ) * damping_amplitude ) + long_distance_prob_limit

    else :

        # Damping from 0 -> limit
        return long_distance_prob_limit * (1. - damping_amplitude)



#
# Decoherence
#

'''
The following models represent different heuristic cases for neutrino - virtual black hole interactions.

They are derived by finding the Lindblad OQS model parameters that match results from the various cases 
using the perturbation model (mapping found using `fit_lindblad_to_toy.py`).

In all cases here, am defining the gamma matrix in the NxN representation.

Note that the NxN representation does support all cases in the SU(N) representation, e.g. in some cases 
multiple SU(N) representation matrices give the same NxN matrix (but different behaviour).
'''

def get_randomize_phase_decoherence_D_matrix(num_states, gamma, basis="sun") :
    '''
    This represents the "randomize_phase" case from the toy model

    A single `gamma` strength parameter defines this, which is found from comparisons
    to the toy model to represent the neutrino BH interaction mean free path.

    The NxN that gives this behaviour is 0 on diagonal and gamma for off-diagonals.
    For SU(N), have gamma for the 11,22,44,55,66,77 elements (zero for all off-diagonals)

    SU(N) to NxN mapping:
        11,22 -> gamma_21
        44,55 -> gamma_32
        66,77 -> gamma_32
        33,88 ("energy-exchange") elements are zero in this case

    This case is also identified in https://arxiv.org/pdf/2001.09250.pdf, e.g. A.12.

    SU(N) :
    [[ 0  0  0  0  0  0  0  0  0]
     [ 0  1  0  0  0  0  0  0  0]
     [ 0  0  1  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  1  0  0  0  0]
     [ 0  0  0  0  0  1  0  0  0]
     [ 0  0  0  0  0  0  1  0  0]
     [ 0  0  0  0  0  0  0  1  0]
     [ 0  0  0  0  0  0  0  0  0]]

    NxN :
    [[0 1 1]
     [1 0 1]
     [1 1 0]]

    '''

    # Toggle based on basis
    if basis == "nxn" :

        D_matrix = np.full( (num_states, num_states), gamma_E, dtype=np.complex128 )
        np.fill_diagonal(D_matrix, 0.)

    elif basis == "sun" :

        num_basis_vectors = np.square(num_states)

        D_matrix = np.diag( [gamma_E]*num_basis_vectors ).astype(np.complex128)

        D_matrix[0,0] = 0.

        assert num_states == 3 #TODO 2 flav
        D_matrix[3,3] = 0.
        D_matrix[8,8] = 0.

    else :
        raise Exception("Unknown basis : %s" % basis)

    return basis, D_matrix


def get_randomize_state_decoherence_D_matrix(num_states, gamma, basis="sun") :
    '''
    This represents the "randomize_state" case from the toy model

    A single `gamma` strength parameter defines this, which is found from comparisons
    to the toy model to represent the neutrino BH interaction mean free path.

    Note that this case is more easily represented in the SU(N) expansion, where it is 
    given by a matrix with `gamma` for all diagonal elements (except the 0'th) and 0 
    for all  off-diagonal elements. 

    SU(N) :
    [[ 0  0  0  0  0  0  0  0  0]
     [ 0  1  0  0  0  0  0  0  0]
     [ 0  0  1  0  0  0  0  0  0]
     [ 0  0  0  1  0  0  0  0  0]
     [ 0  0  0  0  1  0  0  0  0]
     [ 0  0  0  0  0  1  0  0  0]
     [ 0  0  0  0  0  0  1  0  0]
     [ 0  0  0  0  0  0  0  1  0]
     [ 0  0  0  0  0  0  0  0  1]]

    Harder to repsent in NxN here, as the diagonal terms (from the 33/88 SU(N) terms) require mutliple rho elements

    Note that I have checked that I do need the decoherence (non-relaxation) terms as well as the relaxation terms here. 
    '''

    # Basis toggle
    if basis == "nxn" :

        raise NotImplemented

    elif basis == "sun" :

        num_basis_vectors = np.square(num_states)

        D_matrix = np.diag( [gamma]*num_basis_vectors ).astype(np.complex128)

        D_matrix[0,0] = 0.

    else :
        raise Exception("Unknown basis : %s" % basis)

    return basis, D_matrix


def get_neutrino_loss_decoherence_D_matrix(num_states, gamma) :
    '''
    This represents the "neutrino_loss" (non-unitary case from the toy model

    A single `gamma` strength parameter defines this, which is found from comparisons
    to the toy model to represent the neutrino BH interaction mean free path.

    Set all diagonal elements (SU(N)) or all elements (NxN) to a the sterngth value to 
    get this behaviour. Note that for the SU(N) case it is crucial to include 0'th 
    element (for the identiy matrix).

    SU(N) :
    [[ 1  0  0  0  0  0  0  0  0]
     [ 0  1  0  0  0  0  0  0  0]
     [ 0  0  1  0  0  0  0  0  0]
     [ 0  0  0  1  0  0  0  0  0]
     [ 0  0  0  0  1  0  0  0  0]
     [ 0  0  0  0  0  1  0  0  0]
     [ 0  0  0  0  0  0  1  0  0]
     [ 0  0  0  0  0  0  0  1  0]
     [ 0  0  0  0  0  0  0  0  1]]

    NxN :
    [[1 1 1]
     [1 1 1]
     [1 1 1]]
    '''

    # Toggle basis
    basis = "sun"
    if basis == "nxn" :

        D_matrix = np.full( (num_states, num_states), gamma, dtype=np.complex128 )

    elif basis == "sun" :

        num_basis_vectors = np.square(num_states)

        D_matrix = np.diag( [gamma]*num_basis_vectors ).astype(np.complex128)

    else :
        raise Exception("Unknown basis : %s" % basis)


    return basis, D_matrix



def get_vVBH_coherence_length_km_from_gamma_eV(gamma_eV) :
    '''
    TODO ref paper

    Lcoh = 1 / Gamma
    '''

    return natural_to_si_units( 1./(gamma_eV*ureg["eV"]), "length" ).m_as("km")


def get_vVBH_gamma_eV_from_coherence_length_km(Lcoh_km) :
    '''
    TODO ref paper

    Gamma = 1 / Lcoh
    '''

    return ( 1. / si_to_natural_units( Lcoh_km*ureg["km"] ).m_as("1/eV") )




#
# If run this, make a basic set of plots comparing the model
#

if __name__ == "__main__" :

    from analysis.oscNext_decoherence.theory.decoherence_calculator import DecoherenceCalculator
    from analysis.oscNext_decoherence.theory.constants import *

    from utils.script_tools import ScriptWrapper
    from utils.filesys_tools import replace_file_ext
    with ScriptWrapper( replace_file_ext(__file__,".log") ) as script :


        #
        # Create calculator
        #

        num_neutrinos = 2
        gamma_strength_eV = 1.e-14
        gamma_index = 0

        calculator = DecoherenceCalculator(
            tool="dmos",
            atmospheric=False,
            num_neutrinos=num_neutrinos,
            # interactions=False,
        )

        calculator.set_matter("vacuum")


        #
        # Define test cases
        #

        # Standard oscillations
        std_osc_basis = "sun_basis"
        std_osc_D_matrix = np.diag([ 0., 0., 0., 0., 0., 0., 0., 0. ]) if num_neutrinos == 3 else np.diag([ 0., 0., 0. ])

        # Models to test
        model_names = [ "randomize_phase", "neutrino_loss" ] #"randomize_state",  ]

        model_names = ["randomize_state"] #TODO REMOVE


        #
        # Plot
        #

        # Steering
        E = 25.
        L = np.linspace(0., 20.*EARTH_DIAMETER_km, num=1000)
        initial_flavors = [0, 1]
        color_scale = ColorScale("hsv", len(model_names))

        # Loop over initial flavors
        for initial_flavor in initial_flavors :

            fig = Figure( ny=calculator.num_neutrinos+1, sharex=True, figsize=(FIG_WIDTH,7) )

            # Plot std osc
            calculator.set_decoherence_operator_type(std_osc_basis)
            calculator.set_decoherence_D_matrix( D_matrix=std_osc_D_matrix, index=gamma_index )
            calculator.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=E, distance_km=L, color="lightgrey", label="Std osc")

            # Loop over models
            for i_case, model_name in enumerate(model_names) :

                # Set physics params
                basis, D_matrix = get_model_D_matrix(model_name, num_states=num_neutrinos, gamma=gamma_strength_eV)
                calculator.set_decoherence_operator_type(basis)
                calculator.set_decoherence_D_matrix( D_matrix=D_matrix, index=gamma_index )

                # Plot
                calculator.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=E, distance_km=L, color=color_scale.get(i_case), label=model_name)

            # Add long range behaviour lines
            #TODO

            # Format
            # fig.get_ax(y=-1).set_xlabel(DISTANCE_LABEL)
            # fig.quick_format( ylim=(-0.05,1.05) )


        #
        # Done
        #

        print("")
        dump_figures_to_pdf( replace_file_ext(__file__,".pdf") )