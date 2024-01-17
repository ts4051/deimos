'''
Define mathematical functions for working with decoherence in the Lindblad formalism

Tom Stuttard
'''

import numbers

from deimos.utils.constants import PLANCK_MASS_eV, EARTH_DIAMETER_km
from deimos.utils.matrix_algebra import *
# from deimos.utils.oscillations import OSC_FREQUENCY_UNIT_CONVERSION
# from analysis.common.utils.natural_units import si_to_natural_units, natural_to_si_units

# from pisa import ureg

import numpy as np


#
# Decoherence operator maths
#

def get_decoherence_operator_nxn_basis(rho, D_matrix, D_matrix_basis, L=None) :
    '''
    Get the decoherence operator, D[rho]

    In this case, free parameters (gamma) are defined in the same NxN basis as rho is defined 
    in (e.g. standard NxN for N neutrino states matrix that, e.g. no expansion into the SU(N) 
    basis).

    However, am allowing the gamma matrix of free parameters to be defined in the SU(N) basis,
    in which case I convert back to NxN before returning.

    NxN basis just means the standard 3x3 (for 3 neutrino flavor) matrix that 
    matches the density matrix rho, e.g. no expansion into the SU(N) basos
    '''

    #
    # Check inputs
    #

    assert is_square(rho)
    assert is_square(D_matrix)

    num_states = rho.shape[0]


    #
    # Compute D[rho]
    #

    if D_matrix_basis == "nxn" :

        #
        # NxN basis
        #

        '''
        The operator is defined as the element-wise product of rho
        and a "gamma" matrix of real scalars (in the NxN basis), e.g.

        D[rho] = [
            [ rho11*Gamma11, rho12*Gamma12, rho13*Gamma13 ],
            [ rho21*Gamma21, rho22*Gamma22, rho23*Gamma23 ],
            [ rho31*Gamma31, rho32*Gamma32, rho33*Gamma33 ],
        ]
        '''

        # Check inputs
        assert D_matrix.shape == rho.shape

        # Perform element-wise multiplication
        D_rho = np.multiply(rho, D_matrix).astype(np.complex128)


    elif D_matrix_basis == "sun" :

        #
        # SU(N) basis
        #

        '''
        In this case, the gamma matrix of free parameters expressed in the SU(N) basis, 
        e.g. decomposed into coefficients of the basis vectors of SU(N), which is the 
        identity matrix + the generators of SU(N).

        The gamma matrix is (N^2, N^2), although often the 0'th row/column is omitted
        as is always all 0 (handle both cases here).

        The full matrix is extremely general, and has no guarantees of a physical system.
        Various papers show conditions the matrix must fulfill for e.g. complete positivity, 
        energy conservation, increasing entropy, etc, depending on the physics under test.

        In general the off-diagonal components produce more complicated and non-unitary (and 
        possibly CP or CPT violating) effects. A diagonal matrix is a coomon choice in the 
        literature.

        For the diagonal components, the elements can be interpreted as follows (see 2001.09250):
           11/22 - couples to 21 mass splitting (e.g. damps this oscillation frequency, corresponds to gamma21 in NxN case)
           44/55 - couples to 31 mass splitting (e.g. damps this oscillation frequency, corresponds to gamma31 in NxN case)
           66,77 - couples to 32 mass splitting (e.g. damps this oscillation frequency, corresponds to gamma32 in NxN case)
           33,88 - no coupling to oscillation frequencies - referred to as relaxation terms

        Have found that the 33/88 elements produce 1:1:1 long range effects (flavor basis decoherence), 
        but if they are zero then get the averaged oscillation long range case (mass basis decoherence).

        https://arxiv.org/pdf/1711.03680.pdf eqn 7 (2 flavor)
        https://arxiv.org/pdf/2001.07580.pdf eqn 9 (3 flavor)
        '''

        # Check inputs
        sun_num_generators = np.square(num_states) - 1 # e.g. num Dirac/Gell-Mann matrices
        sun_num_basis_vectors = sun_num_generators + 1 # Includes identity

        # If user omitted the 0'th row/col, add it (it is all zeros)
        D_matrix_sun = get_complete_sun_matrix(num_states=num_states, sun_matrix=D_matrix)

        # Physicality checks
        #TODO

        # Decompose rho into SU(N) basis
        rho_sun = convert_matrix_to_sun_decomposition_coefficients(rho)
        assert rho_sun.size == sun_num_basis_vectors

        # Combine gamma and rho to get D[rho], still in the SU(N) basis
        D_rho_sun = np.array([
            np.sum([ ( D_matrix_sun[mu,nu] * rho_sun[nu] ) for nu in range(rho_sun.size) ]) for mu in range(rho_sun.size)
        ], dtype=np.complex128)
        assert D_rho_sun.size ==  sun_num_basis_vectors

        # Re-compose D[rho] into the NxN basis
        D_rho = convert_sun_decomposition_coefficients_to_matrix(D_rho_sun)


    else :
        raise Exception("`gamma_matrix_basis` '%s' not recognised, choose from ['nxn', 'sun']" % gamma_matrix_basis)


    #
    # Distance dependence
    #

    #TODO document

    # if L is not None :
    #     D_rho *= L


    #
    # Done
    #

    assert D_rho.shape == (num_states, num_states)

    return D_rho


def get_decoherence_operator_full_lindblad(
    rho,
    d_operators,
    require_unitarity=False,
    require_increasing_entropy=False,
    require_energy_conservation=False,
) :
    '''
    Get the decoherence operator, D[rho]

    In this case, get the full Lindblad master equation form, no simplicatioms
    '''

    raise NotImplemented("Needs testing/fixing")


    # Check the d operators
    num_states = rho.size[0]
    num_terms = np.square(num_states) - 1
    assert len(d_operators) == num_terms
    for dk in d_operators :
        assert dk.shape == rho.shape

    # Physicality checks, if required
    #TODO Not totally convinced these mathmetical are bullet-proof guarantees to the expected 
    for dk in d_operators :
        if require_unitarity or require_increasing_entropy :
            assert np.allclose(dk, dagger(dk), atol=1.e-20), "dk != dk^dagger :\n%s\n%s" % (dk, dagger(dk)) # This means: dk must be symmetric
        if require_energy_conservation :
            assert np.allclose(commutator(H, dk), np.zeros(dk.shape), atol=1.e-20), "[H,dk] != 0 :\n%s" % dk

    # Compute the operator
    #TODO check the maths here
    #TODO trying two eqautions here from literature that I think are equivalent but seem not to be, why?
    D_rho = np.sum([ anticommutator(rho, np.multiply(dk, dagger(dk))) - 2.*np.multiply(dk, np.multiply(rho, dagger(dk))) for dk in D ], axis=0)
    # D_rho = 0.5 * np.sum([ commutator(dk, np.multiply(rho, dagger(dk))) + commutator(np.multiply(dk, rho), dagger(dk)) for dk in D ], axis=0)

    return D_rho


def get_complete_sun_matrix(num_states, sun_matrix) :
    '''
    SU(N) coefficients usuallly repesented as 8x8 matrix (or 3x3 for 2 flavor), where 8 (3) 
    corresponds to the 8 (3) SU(3/2) generators.

    However, the SU(N) expansion also requires a 0'th element for the identity matrix, which
    is always 0 so generally omitted. Need to add that back in though when actually using
    the gamma matrix to comoute D[rho] (e.g. as a product with the rho coefficients).

    See https://arxiv.org/pdf/1711.03680.pdf equation 7
    '''

    # Check inputs
    assert is_square(sun_matrix)

    # Figure out dimensions
    sun_num_basis_vectors = np.square(num_states)
    sun_num_generators = sun_num_basis_vectors - 1

    # Check what user provided
    if sun_matrix.shape[0] == sun_num_basis_vectors :

        # Is already fine
       return sun_matrix

    elif sun_matrix.shape[0] == sun_num_generators  :

        # User omitted the 0'th row/col, add it (it is all zeros)
        sun_matrix_full = np.zeros((sun_num_basis_vectors, sun_num_basis_vectors), dtype=np.complex128)
        sun_matrix_full[1:,1:] = sun_matrix
        return sun_matrix_full

    else :
        raise Exception("Bad SU(N) matrix shape : %s" % (sun_matrix.shape,) )


def convert_Drho_nxn_to_sun_basis(gamma_matrix) :
    '''
    Convert from the D[rho] defined as an NxN matrix into the corresponding SU(N) expansion.

    See https://arxiv.org/pdf/1711.03680.pdf equation 6.
    '''

    # Check inputs
    assert is_square(gamma_matrix)

    # Perform the decomposition
    gamma_matrix_sun_diag = convert_matrix_to_sun_decomposition_coefficients(gamma_matrix)

    # Remove the all-zero 0'th element correspondong to the identity matrix (this is a convention)
    gamma_matrix_sun_diag = gamma_matrix_sun_diag[1:]

    # Format the matrix
    gamma_matrix_sun = np.diag(gamma_matrix_sun_diag)

    return gamma_matrix_sun


def convert_Drho_sun_to_nxn_basis(gamma_matrix) :
    '''
    Convert from the D[rho] defined as an SU(N) expansion to more straight forward NxN representation in the mass basis.

    See https://arxiv.org/pdf/1711.03680.pdf equation 6.

    https://arxiv.org/pdf/2001.09250.pdf A.12 also defines this particular mapping for themselves.
    '''

    # Check inputs
    assert is_square(gamma_matrix)

    # Check inputs and get the num neutrino states
    gamma_matrix_dim = gamma_matrix.shape[0]
    if gamma_matrix_dim == 3 :
        num_neutrinos = 2
    elif gamma_matrix_dim == 8 :
        num_neutrinos = 3

    # Check no off-diagonal non-zero elements - implementation is restricted to diagonals
    #TODO is this true?
    diag_mask = np.identity(gamma_matrix.shape[0]).astype(bool)
    off_diag_mask = np.logical_xor(np.ones(shape=gamma_matrix.shape, dtype=bool), diag_mask)
    assert np.allclose(gamma_matrix[off_diag_mask], 0.), "Found non-zero off-diagonal elements"

    # Add the 0'th element corresponding to the identity matrix
    gamma_matrix_diag = np.diag(gamma_matrix)
    gamma_matrix_diag = np.array([0.] + gamma_matrix_diag.tolist())

    # Re-compose the coefficients into the NxN matrix
    return convert_sun_decomposition_coefficients_to_matrix(gamma_matrix_diag)


def get_decoherence_operator_nxn_basis_from_sun_D_matrix(D_matrix, rho) :
    '''
    Define the NxN decoherence operator using the SU(N) D matrix
    This is an analytical mapping, and is used to verify the derivation in deoch paper eqns 14/15 
    '''

    # Currently only supporting 3 flavor
    num_states = rho.shape[0]
    assert num_states == 3

    # rho should be NxN
    assert rho.shape == (num_states, num_states)

    # D should be SU(N)
    assert D_matrix.shape == (num_states**2, num_states**2)

    # Currently only supporting diagonal D matrix
    #TODO check

    Gamma_0 = D_matrix[0,0]
    Gamma_1 = D_matrix[1,1]
    Gamma_2 = D_matrix[2,2]
    Gamma_3 = D_matrix[3,3]
    Gamma_4 = D_matrix[4,4]
    Gamma_5 = D_matrix[5,5]
    Gamma_6 = D_matrix[6,6]
    Gamma_7 = D_matrix[7,7]
    Gamma_8 = D_matrix[8,8]

    Omega_0 = ( Gamma_0 / 3. ) * (rho[0,0] + rho[1,1] + rho[2,2])
    Omega_3 = ( Gamma_3 / 2. ) * (rho[0,0] - rho[1,1])
    Omega_8 = ( Gamma_8 / 6. ) * (rho[0,0] + rho[1,1] - 2.*rho[2,2])

    Drho_00 = Omega_0 + Omega_3 + Omega_8
    Drho_11 = Omega_0 - Omega_3 + Omega_8
    Drho_22 = Omega_0 - 2.*Omega_8

    Drho_01 = ( rho[0,1].real * Gamma_1 ) - 1j * ( rho[1,0].imag * Gamma_2 )
    Drho_10 = ( rho[0,1].real * Gamma_1 ) + 1j * ( rho[1,0].imag * Gamma_2 )

    Drho_02 = ( rho[0,2].real * Gamma_4 ) - 1j * ( rho[2,0].imag * Gamma_5 )
    Drho_20 = ( rho[0,2].real * Gamma_4 ) + 1j * ( rho[2,0].imag * Gamma_5 )

    Drho_12 = ( rho[1,2].real * Gamma_6 ) - 1j * ( rho[2,1].imag * Gamma_7 )
    Drho_21 = ( rho[1,2].real * Gamma_6 ) + 1j * ( rho[2,1].imag * Gamma_7 )

    Drho = np.array([
        [ Drho_00, Drho_01, Drho_02 ],
        [ Drho_10, Drho_11, Drho_12 ],
        [ Drho_20, Drho_21, Drho_22 ],
    ], dtype=np.complex128)

    return Drho



#
# Testing
#

# Define a rho for testing
# This is a random one grabbed from the solver, which is valid (Hermitian) and has many non-zero elements
TEST_RHO_NxN = np.array([
        [ 0.21276528+0.00000000e+00j, -0.22456552-5.25078607e-03j, -0.23842018-2.45346247e-01j],
        [-0.22456552+5.25078607e-03j, 0.23714979+0.00000000e+00j, 0.25769812+2.53069548e-01j],
        [-0.23842018+2.45346247e-01j, 0.25769812-2.53069548e-01j, 0.55008493+0.00000000e+00j],
    ], dtype=np.complex128)


def generate_test_density_matrix_mass_basis(random_state=None) :
    '''
    Generate a random density matrix with complex non-zero elements.
    Will be a valid Hermitian matrix.

    Useful for testing.
    '''

    from deimos.utils.density_matrix_osc_solver.density_matrix_osc_solver import get_rho, get_pmns_matrix, rho_flav_to_mass_basis, dagger

    if random_state is None :
        random_state = np.random.RandomState(12345)

    # Generate a density matrix in the flav basis from a random linear combination of flavors
    norm = 10.
    rho_flav = random_state.uniform(0., norm) * get_rho([1., 0., 0.])
    rho_flav += random_state.uniform(0., norm) * get_rho([0., 1., 0.])
    rho_flav += random_state.uniform(0., norm) * get_rho([0., 0., 1.])

    # Generate a random PMNS
    pmns = get_pmns_matrix( theta=random_state.uniform(0., np.pi/2., size=3), dcp=random_state.uniform(0., 2.*np.pi) )

    # Rotate to mass basis
    rho_mass = rho_flav_to_mass_basis(rho_flav, pmns)

    # Check if hermitian
    assert np.all( np.isclose(rho_mass, dagger(rho_mass)) )

    return rho_mass


def test_decoherence_operator_functions() :

    random_state = np.random.RandomState(12345)

    #
    # Generate a random density matrix, many times
    #

    for i in range(1000) :

        rho = generate_test_density_matrix_mass_basis(random_state=random_state)

        # Check some basic properties of the system
        for i, j in [ [0, 1], [0, 2], [1, 2]] :
            assert np.isclose(rho[i,j].real, rho[j,i].real), "rho[%i,%i].real != rho[%i,%i].real (%0.3g != %0.3g)" % (i,j, j,i, rho[i,j].real, rho[j,i].real)
            assert np.isclose(rho[i,j].imag, -rho[j,i].imag), "rho[%i,%i].imag != -rho[%i,%i].imag (%0.3g != %0.3g)" % (i,j, j,i, rho[i,j].imag, rho[j,i].imag)



        #
        # Compute D[rho] in the NxN basis given a SU(N) D matrix, and comapre results from different methods
        #

        for i in range(10) :

            # Generate random diagonal D matrices
            # Note that these are not in general physical, but still fine for testing
            D_matrix = np.diag( random_state.uniform(0., 1., size=9) ).astype(np.complex128)

            # Calc D[rho] using the main, general functions I have written
            Drho_main = get_decoherence_operator_nxn_basis(rho=rho, D_matrix=D_matrix, D_matrix_basis="sun", L=None)

            # For comparison, compute D[rho] using the 3-flavor derived form from my paper
            Drho_derived = get_decoherence_operator_nxn_basis_from_sun_D_matrix(D_matrix=D_matrix, rho=rho)

            # Compare
            for idx in np.ndindex(Drho_main.shape) :
                # print("%s -> %s vs %s" % (idx, Drho_main[idx], Drho_derived[idx]))
                assert np.isclose(Drho_main[idx], Drho_derived[idx])


        #
        # Specifically check phase peturbation mapping case
        #

        Gamma = 123.

        D_matrix = np.diag( [0., Gamma, Gamma, 0., Gamma, Gamma, Gamma, Gamma, 0.] ).astype(np.complex128)

        Gamma_21 = Gamma
        Gamma_31 = Gamma
        Gamma_32 = Gamma
        Drho_nxn = np.array([
            [ 0., Gamma_21*rho[0,1], Gamma_31*rho[0,2] ],
            [ Gamma_21*rho[1,0], 0., Gamma_31*rho[1,2] ],
            [ Gamma_21*rho[2,0], Gamma_31*rho[2,1], 0. ],
        ], dtype=np.complex128)

        Drho_from_sun = get_decoherence_operator_nxn_basis_from_sun_D_matrix(D_matrix=D_matrix, rho=rho)

        # Compare
        for idx in np.ndindex(Drho_main.shape) :
            # print("%s -> %s vs %s" % (idx, Drho_main[idx], Drho_derived[idx]))
            assert np.isclose(Drho_main[idx], Drho_derived[idx])



def check_decoherence_D_matrix(num_neutrinos, D) :
    '''
    There exist inequalities between the elements of the decoherence D matrix, meaning that the elements are not fully independent

    Enforcing these inequalities here, as defined in:
     - 2 flavor: https://arxiv.org/pdf/hep-ph/0105303.pdf
     - 3 flavor: https://arxiv.org/pdf/1811.04982.pdf Appendix B
    '''

    #TODO implement the 2nu constraints
    
    if num_neutrinos == 3 :

        #
        # SU(3) case
        #

        #TODO What enforces g1=g1, g4=g5, g6=g7 ?

        assert D.shape == (9, 9)

        #TODO what about 0th row/col?

        # Check everything is real
        assert np.all( D.imag == 0. )

        # Check everything is positive or zero
        assert np.all( D >= 0. )

        # Extract diagonal elements (gamma)
        g1 = D[1,1]
        g2 = D[2,2]
        g3 = D[3,3]
        g4 = D[4,4]
        g5 = D[5,5]
        g6 = D[6,6]
        g7 = D[7,7]
        g8 = D[8,8]

        # Extract off-diagonal elements (beta)
        # Enforce pairs either side of the diagonal match in the process
        b12 = D[1,2]
        assert D[2,1] == b12
        b13 = D[1,3]
        assert D[3,1] == b13
        b14 = D[1,4]
        assert D[4,1] == b14
        b15 = D[1,5]
        assert D[5,1] == b15
        b16 = D[1,6]
        assert D[6,1] == b16
        b17 = D[1,7]
        assert D[7,1] == b17
        b18 = D[1,8]
        assert D[8,1] == b18
        b23 = D[2,3]
        assert D[3,2] == b23
        b24 = D[2,4]
        assert D[4,2] == b24
        b25 = D[2,5]
        assert D[5,2] == b25
        b26 = D[2,6]
        assert D[6,2] == b26
        b27 = D[2,7]
        assert D[7,2] == b27
        b28 = D[2,8]
        assert D[8,2] == b28
        b34 = D[3,4]
        assert D[4,3] == b34
        b35 = D[3,5]
        assert D[5,3] == b35
        b36 = D[3,6]
        assert D[6,3] == b36
        b37 = D[3,7]
        assert D[7,3] == b37
        b38 = D[3,8]
        assert D[8,3] == b38
        b45 = D[4,5]
        assert D[5,4] == b45
        b46 = D[4,6]
        assert D[6,4] == b46
        b47 = D[4,7]
        assert D[7,4] == b47
        b48 = D[4,8]
        assert D[8,4] == b48
        b56 = D[5,6]
        assert D[6,5] == b56
        b57 = D[5,7]
        assert D[7,5] == b57
        b58 = D[5,8]
        assert D[8,5] == b58
        b67 = D[6,7]
        assert D[7,6] == b67
        b68 = D[6,8]
        assert D[8,6] == b68
        b78 = D[7,8]
        assert D[8,7] == b78

        # Now implement all inequalities
        a1 = -g1 + g2 + g3 - (g8/3.) 
        a2 =  g1 - g2 + g3 - (g8/3.)
        a3 =  g1 + g2  -g3 - (g8/3.)

        a4 = -g4 + g5 + g3 + (2.*g8/3.) - (2.*b38/np.sqrt(3.)) # See here that beta38 is somehwat special (since it relates to the special gamma3/8 params)
        a5 =  g4 - g5 + g3 + (2.*g8/3.) - (2.*b38/np.sqrt(3.))
        a6 = -g6 + g7 + g3 + (2.*g8/3.) + (2.*b38/np.sqrt(3.))
        a7 =  g6 - g7 + g3 + (2.*g8/3.) + (2.*b38/np.sqrt(3.))

        a8 = -(g1/3.) - (g2/3.) - (g3/3.) + (2.*g4/3.) + (2.*g5/3.) + (2.*g6/3.) + (2.*g7/3.) - g8

        assert a1 >= 0., "Inequality failure (a1)"
        assert a2 >= 0., "Inequality failure (a2)"
        assert a3 >= 0., "Inequality failure (a3)"
        assert a4 >= 0., "Inequality failure (a4)"
        assert a5 >= 0., "Inequality failure (a5)"
        assert a6 >= 0., "Inequality failure (a1)"
        assert a7 >= 0., "Inequality failure (a7)"
        assert a8 >= 0., "Inequality failure (a8)"

        assert (4.*np.square(b12)) <= ( np.square(g3 - (g8/3.)) - np.square(g1 - g2) )
        assert (4.*np.square(b13)) <= ( np.square(g2 - (g8/3.)) - np.square(g1 - g3) )
        assert (4.*np.square(b23)) <= ( np.square(g1 - (g8/3.)) - np.square(g2 - g3) )

        assert np.square( 4.*np.square(b38) + (g4/np.sqrt(3.)) + (g5/np.sqrt(3.)) - (g6/np.sqrt(3.)) - (g7/np.sqrt(3.)) ) <= (a3*a8)

        #TODO there are still quite a few more involving beta....


    else :

        # Error handling
        raise NotImplementedError("Checks on decoherence D matrix inequalities not yet implemented for a %i neutrino system" % num_neutrinos)




#
# Interface to decoherence models
#

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

    else :
        raise Exception("Unknown model : %s" % model_name)




#
# Simple oscillation calculation
#

#TODO factor out into another script?

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
# Space-time metric
#

# Define relevent metrics
FLAT_SPACETIME_METRIC_TENSOR = np.array([ # Minkowski
    [ -1., 0., 0., 0., ], #TODO c ?
    [  0., 1., 0., 0., ],
    [  0., 0., 1., 0., ],
    [  0., 0., 0., 1., ],
])

def get_ds_from_metric_tensor(dx, g=FLAT_SPACETIME_METRIC_TENSOR) :
    '''
    Compute the distance/displacement step, ds, corresponding to some time/spatial dimension steps, dx, 
    for a given space-time metric tensor, g.

    For 4-D space-time, dx = [dt, dx, dy, dz], and the metric tensor is 4x4

    See https://www.mv.helsinki.fi/home/syrasane/cosmo/lect2018_02.pdf

    Note that off-diagonal metric tensor components represent non-orthogonal coordinate systems,
    which are not common so rarely use these.
    '''

    #TODO also do integral to get overall distance here?

    dim = dx.size
    assert g.shape == (dim, dim)

    ds2 = 0.
    for mu in range(dim) :
        for nu in range(dim) :
            ds2 += ( g[mu, nu] * dx[mu] * dx[nu] )

    return np.sqrt(ds2)


def get_fluctuated_metric_tensor(a1, a2, a3, a4) :
    '''
    Using the definition of metric fluctuations From hep-ph/0606048

    Specificially eqn 2.4 (derived from 2.2, 2.3)

    This is 1+1D ([t, x]) metric tensor with fluctuations characterised by static coefficients ai that are Gaussian random variables with <ai> = 0
    Can choose x to lie along particle direction
    
    ai are Gaussian random variables with <ai> = 0 and sigmai

    a4 characterises the distance-only flucutation

    Note that the analytic g expression below is a fluctuation of flat space-time (e.g. Minkowski metric tensor). Could in princip
    '''

    # Check that fluctuations are small or at least comaprable to the overall metric structure
    # This is an assumption used in this model (and a very reasonable one)
    # If have e.g. a4 < -1, ds starts to rise again  with decreasing a4 due to the sqrt( (a4+1)^2 ) term, which is
    # clearly nonsense (and thus a limitations of the parameterisation)
    assert np.all( np.abs(a1) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"
    assert np.all( np.abs(a2) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"
    assert np.all( np.abs(a3) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"
    assert np.all( np.abs(a4) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"

    # Define metric
    # This is specifically for flat space-time + perturbation
    g = np.array([
        [    ( -1. * np.square(a1 + 1) ) + np.square(a2)     ,     ( -1. * a3 * (a1 + 1) ) +  ( a2 * (a4 + 1) )    ],
        [    ( -1. * a3 * (a1 + 1) ) +  ( a2 * (a4 + 1) )    ,     ( -1. * np.square(a3) ) +  np.square(a4 + 1)    ],
    ])


    # Optionally, verify the derivation (see eqns 2.2 and 2.3)
    if False :

        O = np.array([ # This is the fundamental perurbation
            [ a1+1., a2, ],
            [ a3, a4+1, ],
        ])

        eta = np.array([ # Note: Could use a different metric here if desired (this is flat space-time)
            [ -1., 0., ],
            [ 0., 1., ],
        ])

        g_v2 = np.matmul( O, np.matmul(eta, O.T) )

        assert np.array_equal(g, g_v2)

    # Done
    return g


#
# Model implementations
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



#
# Test
#

if __name__ == "__main__" :


    # #
    # # Test conversion between phenomenological gamma_0 and Planck scale lambda expressions
    # #

    # # 99% limits from hep-ph/0506168 (eqn 22-26)
    # # In the paper's notation: gamma_0->kappa, lambda->kappatilda
    # cases_n_gamma_0_GeV = [
    #     ( -1., 2.3e-31 ),
    #     ( 0., 3.1e-34 ),
    #     ( 1., 7.2e-39 ),
    #     ( 2., 5.5e-42 ),
    #     ( 3., 2.9e-45 ),
    # ]

    # # Calculate
    # # Only available reference value from the paper is the n=3 case, which is lambda=3e-7
    # #TODO I get 4.3e-07 for n=3, not 3e-7. Why??
    # print("\nExamples from hep-ph/0506168 (using 99% limits) :")
    # for n, gamma_0_GeV in cases_n_gamma_0_GeV :
    #     gamma_0_inv_m = convert_gamma_eV_to_gamma_inv_m(gamma_0_GeV*1.e9)
    #     lambda_Planck = convert_gamma0_to_lambda_Planck( gamma0_eV=(gamma_0_GeV*1.e9), E0_eV=1.e9, n=n )
    #     print(("  n = %0.3g : gamma_0 = %0.3g GeV : 1/gamma_0 = %0.3g m : lambda_Planck = %0.3g" % (n, gamma_0_GeV,gamma_0_inv_m,lambda_Planck)))
    # print("")


    #
    # Demonstrate gamma -> 1/gamma conversion
    #

    # print("\n1/Gamma = 1 Earth diameter -> Gamma = %0.3g eV" % convert_gamma_inv_m_to_gamma_eV(EARTH_DIAMETER_km*1e3) )

    # print( convert_gamma_eV_to_gamma_inv_m(0.6e-15)*1e-3/EARTH_DIAMETER_km )
    # print( convert_gamma_eV_to_gamma_inv_m(2.e-15)*1e-3/EARTH_DIAMETER_km )
    # print( convert_gamma_eV_to_gamma_inv_m(8.e-15)*1e-3/EARTH_DIAMETER_km )


    #
    # Test decoherence operator maths
    #

    test_decoherence_operator_functions()
    
