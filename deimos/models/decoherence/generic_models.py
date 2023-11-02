'''
Implemeting generic decoherence models, e.g. D matrix textures without specific assumptions about the underlying microphysics

Tom Stuttard
'''

import numpy as np

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
