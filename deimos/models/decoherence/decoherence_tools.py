'''
Miscellaneous functions/tools for working with neutrino decoherence 
Tom Stuttard
'''

#
# Analytic expression
#

def calc_osc_prob(PMNS, mass_splittings_eV2, L_km, E_GeV, initial_flavor, final_flavor, Lcoh_km=None, L_index=None) :
    '''
    Oscillation probability calculation, with decoherence

    https://en.wikipedia.org/wiki/Neutrino_oscillation

    https://arxiv.org/pdf/1805.09818.pdf eqn 14
    '''

    raise Exception("Doesn't quite give the correct frequency (although pretty close), needs debugging")

    # Checks
    num_states = PMNS.shape[0]
    assert num_states in [2, 3]
    assert is_square(PMNS)
    assert mass_splittings_eV2.size == ( 1 if num_states == 2 else 3)

    # Units
    E = si_to_natural_units( E_GeV*ureg["GeV"] ).m_as("eV")
    L = si_to_natural_units( L_km*ureg["km"] ).m_as("1/eV")
    if Lcoh_km is not None :
        Lcoh = si_to_natural_units( Lcoh_km*ureg["km"] ).m_as("1/eV")
        assert L_index is not None

    # Do the calc...
    kronecker_delta = float(initial_flavor == final_flavor)
    osc_prob = kronecker_delta

    for i in range(num_states) :

        for j in range(i+1, PMNS.shape[0]) :

            PMNS_product = np.conj(PMNS[initial_flavor,i]) * PMNS[final_flavor,i] * PMNS[initial_flavor,j] * np.conj(PMNS[final_flavor,j])

            frequency_term = (mass_splittings_eV2[i] * L) / (2. * E)
            # frequency_term = (2. * OSC_FREQUENCY_UNIT_CONVERSION * mass_splittings_eV2[i] * L_km / E_GeV)
            
            if Lcoh_km is not None :
                damping_term = np.exp( -1. * ( (L/Lcoh[i]) )**(2.*L_index) ) #TODO not working, fix this...
            else :
                damping_term = 1.

            osc_prob_term = -2. * np.real(PMNS_product)

            osc_prob_term += 2. * damping_term * np.real(PMNS_product) * np.cos(frequency_term)
            
            osc_prob_term += 2. * damping_term * np.imag(PMNS_product) * np.sin(frequency_term)

            osc_prob += osc_prob_term

    
    return osc_prob



#
# Model interface
#

def get_generic_model_decoherence_D_matrix(name, gamma) :
    '''
    Return the D matrix for a given generic model, using the definitions in https://arxiv.org/abs/2306.14699
    '''

    #
    # Get the texture
    #

    if name == "A" :
        gamma21 = gamma
        gamma31 = gamma
        gamma32 = gamma

    elif name == "B" :
        gamma21 = gamma
        gamma31 = gamma
        gamma32 = 0.

    elif name == "C" :
        gamma21 = gamma
        gamma31 = 0.
        gamma32 = gamma

    elif name == "D" :
        gamma21 = 0.
        gamma31 = gamma
        gamma32 = gamma

    elif name == "E" :
        gamma21 = gamma
        gamma31 = 0.
        gamma32 = 0.

    elif name == "F" :
        gamma21 = 0.
        gamma31 = gamma
        gamma32 = 0.

    elif name == "G" :
        gamma21 = 0.
        gamma31 = 0.
        gamma32 = gamma

    else :
        raise Exception("Unknown model")


    #
    # Form the D matrix
    #

    D = np.diag([0., gamma21, gamma21, 0., gamma31, gamma31, gamma32, gamma32, 0.])

    return D
    

def get_model_D_matrix(model_name, num_states, **kw) :
    '''
    Top-level function to get one of the models defined in this script using a string name
    '''

    #TODO Not currenrly consistent between models about how E-dependence is implemented (e.g. within the D matrix here, or externally). Make this more consistent once have figured out the best way to implement lightcone fluctuations in nuSQuIDS

    from deimos.utils.model.nuVBH_interactions.nuVBH_model import get_randomize_phase_decoherence_D_matrix, get_randomize_state_decoherence_D_matrix, get_neutrino_loss_decoherence_D_matrix
    from deimos.utils.model.lightcone_fluctuations.lightcone_fluctuation_model import get_lightcone_decoherence_D_matrix

    if model_name == "randomize_phase" :
        return get_randomize_phase_decoherence_D_matrix(num_states=num_states, **kw)

    elif model_name == "randomize_state" :
        return get_randomize_state_decoherence_D_matrix(num_states=num_states, **kw)

    elif model_name == "neutrino_loss" :
        return get_neutrino_loss_decoherence_D_matrix(num_states=num_states, **kw)

    elif model_name == "lightcone" :
        return get_lightcone_decoherence_D_matrix(num_states=num_states, **kw)

    #TODO integrate get_generic_model_decoherence_D_matrix

    else :
        raise Exception("Unknown model : %s" % model_name)

