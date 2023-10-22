'''
Matrix algebra
Specifically algebra required for quantum mechanics, in particular neutrino oscillations

Tom Stuttard
'''

import numpy as np
import copy


#
# Basic matrix operations
#

def is_square(matrix) :
    """ Check if matrix is a square matrix, e.g. NxN """
    return ( matrix.ndim == 2 ) and ( matrix.shape[0] == matrix.shape[1] )


def is_unitary(matrix) :
    """ Check matrix is unitary, e.g. U U^dagger = 1 """
    #TODO Check NxN
    assert is_square(matrix)
    return np.allclose( np.dot( matrix, dagger(matrix) ), np.diagflat([1.]*matrix.shape[0]).astype(np.complex128) )


def is_hermitian(matrix) :
    """ Check matrix is Hermitian, e.g. U = U^dagger """
    assert is_square(matrix)
    return np.allclose( matrix, dagger(matrix) )


def dagger(matrix) :
    """ Apply 'dagger' operation to matrix, e.g. transpose + complex conjugate """
    return np.conj(matrix.T)

def conj_transpose(matrix) : # Alias
    return dagger(matrix)


def commutator(A,B) : 
    """Quantum mechanics commutator: [A,B] = AB - BA"""
    return ( np.dot(A,B) - np.dot(B,A) ).astype(np.complex128)


def anticommutator(A,B) : 
    """Quantum mechanics anticommutator: {A,B} = AB + BA"""
    return ( np.dot(A,B) + np.dot(B,A) ).astype(np.complex128)

def is_real(A) :
    '''Check all elemens are real''' 
    return np.allclose(A.imag, 0.)


#
# SU(N) algebra
#

'''
This section involes re-expressing NxN Hermitian matrices as linear combinations of the generators of SU(N).
This can be used in neutrino oscillation equations.

References :
  [1] https://arxiv.org/pdf/1904.12391.pdf
  [2] https://arxiv.org/pdf/1412.3832.pdf
  [3] https://arxiv.org/pdf/2001.07580.pdf
'''

#TODO wrapper function for SU(N)

# SU(2) generators are the Pauli matrices
PAULI_MATRICES = [
    np.array( [ [0,   1], [ 1,  0] ], dtype=np.complex128 ),
    np.array( [ [0, -1j], [1j,  0] ], dtype=np.complex128 ),
    np.array( [ [1,   0], [ 0, -1] ], dtype=np.complex128 ),
]

SU2_GENERATORS = [
    np.array( [ [1, 0], [0, 1] ], dtype=np.complex128 ), # Identity matrix
    PAULI_MATRICES[0],
    PAULI_MATRICES[1],
    PAULI_MATRICES[2],
]

# SU(3) generators are the Gell-Mann matrices
GELL_MANN_MATRICES = [
    np.array( [ [0,   1,   0], [ 1,  0,   0], [ 0,  0,  0] ], dtype=np.complex128 ), #               [1] in nuSQuIDS generators convention
    np.array( [ [0, -1j,   0], [ 1j, 0,   0], [ 0,  0,  0] ], dtype=np.complex128 ), #              -[3] in nuSQuIDS generators convention
    np.array( [ [1,   0,   0], [ 0, -1,   0], [ 0,  0,  0] ], dtype=np.complex128 ), #               [4] in nuSQuIDS generators convention
    np.array( [ [0,   0,   1], [ 0,  0,   0], [ 1,  0,  0] ], dtype=np.complex128 ), #               [2] in nuSQuIDS generators convention
    np.array( [ [0,   0, -1j], [ 0,  0,   0], [1j,  0,  0] ], dtype=np.complex128 ), #              -[6] in nuSQuIDS generators convention
    np.array( [ [0,   0,   0], [ 0,  0,   1], [ 0,  1,  0] ], dtype=np.complex128 ), #               [5] in nuSQuIDS generators convention
    np.array( [ [0,   0,   0], [ 0,  0, -1j], [ 0, 1j,  0] ], dtype=np.complex128 ), #              -[7] in nuSQuIDS generators convention
    np.array( [ [1,   0,   0], [ 0,  1,   0], [ 0,  0, -2] ], dtype=np.complex128 ) / np.sqrt(3.), # [8] in nuSQuIDS generators convention
]

SU3_GENERATORS = [
    np.array( [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ], dtype=np.complex128 ), # Identity matrix
    GELL_MANN_MATRICES[0],
    GELL_MANN_MATRICES[1],
    GELL_MANN_MATRICES[2],
    GELL_MANN_MATRICES[3],
    GELL_MANN_MATRICES[4],
    GELL_MANN_MATRICES[5],
    GELL_MANN_MATRICES[6],
    GELL_MANN_MATRICES[7],
]


# Define structure constants
# SU3_STRUCTURE_CONSTANTS = np.zeros((9,9,9), dtype=np.float64)
# SU3_STRUCTURE_CONSTANTS[1,2,3] = 1.
# SU3_STRUCTURE_CONSTANTS[1,4,7] = 1. / 2.
# SU3_STRUCTURE_CONSTANTS[1,6,5] = SU3_STRUCTURE_CONSTANTS[1,4,7]
# SU3_STRUCTURE_CONSTANTS[2,4,6] = SU3_STRUCTURE_CONSTANTS[1,4,7]
# SU3_STRUCTURE_CONSTANTS[2,5,7] = SU3_STRUCTURE_CONSTANTS[1,4,7]
# SU3_STRUCTURE_CONSTANTS[3,4,5] = SU3_STRUCTURE_CONSTANTS[1,4,7]
# SU3_STRUCTURE_CONSTANTS[3,7,6] = SU3_STRUCTURE_CONSTANTS[1,4,7]
# SU3_STRUCTURE_CONSTANTS[4,5,8] = np.sqrt(3.) / 2.
# SU3_STRUCTURE_CONSTANTS[6,7,8] = SU3_STRUCTURE_CONSTANTS[4,5,8]

# #TODO Not sure this is working
# def fill_out_antisymmetric_matrix(array) :
#     for index in np.ndindex(array.shape) :
#         array[(index[0], index[2], index[1])] = -1*array[index]
#         array[(index[2], index[1], index[0])] = -1*array[index]
#         array[(index[1], index[0], index[2])] = -1*array[index]
#     assert np.array_equal(array, -1.*array.T), "Matrix is not antisymmetric"
# fill_out_antisymmetric_matrix(SU3_STRUCTURE_CONSTANTS) 


def convert_matrix_to_su2_decomposition_coefficients(matrix) :
    '''
    Decompose a 2x2 square matrix to a linear combination of SU(2) generators (Pauli matrices + identity)
    
    Matrix here can be Hamiltonian, density matrix, etc
    Any 2x2 Hermitian matrix can be expanded in terms of SU(2) generators, since these 
    repreent the basis set (e.g. orthogonal axes) for the vector space

    *I think* the SU(2) coefficients are guaranteed to be real

    The expression for the coefficients is taken from ref [1] Table I
    TODO would be nice to directly use the Pauli matrix definitions if possible

    Returns the coefficients of each generator
    '''

    assert is_hermitian(matrix), "matrix must be Hermitian"

    return np.array([
        0.5 * ( matrix[0,0] + matrix[1,1] ),
        matrix[0,1].real,
        -1. * matrix[1,0].imag,
        0.5 * ( matrix[0,0] - matrix[1,1] ),
    ], dtype=np.complex128 )


def convert_matrix_to_su2_decomposition_coefficients_nusquids(matrix) :
    '''
    This is doing the same thing as `convert_matrix_to_su2_decomposition_coefficients`,
    but using equations copied from SQuIDS (MatrixToSU2.txt).

    In this case it is identifcal to my own one using the Pauli matrices (not the case for SU(3))
    '''

    assert is_hermitian(matrix), "matrix must be Hermitian"

    coeffts = np.array([
        0.5 * ( matrix[0,0] + matrix[1,1].real ),
        0.5 * ( matrix[0,1].real + matrix[1,0].real ),
        0.5 * ( matrix[1,0].imag - matrix[0,1].imag ),
        0.5 * ( matrix[0,0].real - matrix[1,1].real ),
    ], dtype=np.complex128 )

    assert is_real(coeffts), "coefficients must be real"

    return coeffts



def convert_matrix_to_su3_decomposition_coefficients(matrix) :
    '''
    Decompose a 3x3 square matrix to a linear combination of SU(3) generators (Gell-Mann matrices + identity)
    
    Read the documentation for `convert_matrix_to_su2_decomposition_coefficients` for more details

    Returns the coefficients of each generator

    See [3] eqn 13 for an example of the decomposed density matrix

    WARNING: This is correct for the definition of the Gell-Mann matrices in this code (same as on wikipedia)
             SQuIDS uses a differnt but equivalent convention (better suited to SU(N) generalisation), but cannot 
             mix the two.

    Extra notes:
    ------------
    My hand-derived SU(3) -> 3x3 conversion:
        r00 = r0 + r3 + r8/sqrt(3)
        r01 = r1 - r2j
        r02 = r4 - r5j
        r10 = r1 + r2j
        r11 = r0 - r3 + r8/sqrt(3)
        r12 = r6 - r7j
        r20 = r4 + r5j
        r21 = r6 + r7j
        r22 = r0 - 2r8/sqrt(3)

    My version of this mapping: TODO fully derive, here just matching elements
        r0 -> [r00, r11, r22]
        r1 -> [r01, r10]
        r2 -> [r01, r10]
        r3 -> [r00, r11]
        r4 -> [r02, r20]
        r5 -> [r02, r20]
        r6 -> [r12, r21]
        r7 -> [r12, r21]
        r8 -> [r00, r11, r22]

    '''

    assert is_hermitian(matrix), "matrix must be Hermitian"

    # This version is from ref [1] Table II
    #TODO in some cases this assumes reflection symmetry about the diagonal (e.g. [1,0]=[0,1]), generalise this like nuSQuIDS does
    coeffts = np.array([
        ( matrix[0,0] + matrix[1,1] + matrix[2,2] ) / 3.,
        matrix[0,1].real,
        -1. * matrix[0,1].imag,
        ( matrix[0,0] - matrix[1,1] ) / 2.,
        matrix[0,2].real,
        -1. * matrix[0,2].imag,
        matrix[1,2].real,
        -1. * matrix[1,2].imag,
        ( matrix[0,0] + matrix[1,1] - (2. * matrix[2,2]) ) * np.sqrt(3.) / 6.,
    ], dtype=np.complex128 )

    assert is_real(coeffts), "coefficients must be real"

    return coeffts


def convert_matrix_to_su3_decomposition_coefficients_nusquids(matrix) :
    '''
    This is the version copied from SQuIDS

    Note that this DOESN'T match my SU(3) version above due the difference in convention
    '''

    assert is_hermitian(matrix), "matrix must be Hermitian"

    # nuSQuIDS original
    coeffts = np.array([
        ( matrix.real[0][0] + matrix.real[1][1] + matrix.real[2][2] ) / 3.,
        ( matrix.real[0][1] + matrix.real[1][0] ) / 2.
        ( matrix.real[0][2] + matrix.real[2][0] ) / 2.,
        ( -matrix.imag[0][1] + matrix.imag[1][0] ) / 2.,
        ( matrix.real[0][0] + -matrix.real[1][1]) / 2.,
        ( matrix.real[1][2] + matrix.real[2][1] ) / 2.,
        ( -matrix.imag[0][2] + matrix.imag[2][0] ),
        ( -matrix.imag[1][2] + matrix.imag[2][1] ),
        1/(2.*np.sqrt(3))*matrix.real[0][0] + 1/(2.*np.sqrt(3))*matrix.real[1][1] + (1/np.sqrt(3))*matrix.real[2][2],
    ], dtype=np.complex128 )


    assert is_real(coeffts), "coefficients must be real"

    return coeffts


def convert_su2_decomposition_coefficients_to_matrix(su2_decomposition_coefficients) :
    '''
    Re-compose the linear combination of SU(2) generators representation of a 2x2 square matrix back into a 2X2 square matrix
    
    Read the documentation for `convert_matrix_to_su2_decomposition_coefficients` for more details

    Basically just multiply the each coefficient by the associated generator (reg [2], equation 15)
    '''

    assert is_real(su2_decomposition_coefficients), "coefficients must be real"

    # Here I directly use the SU(2) generators according to 
    matrix = sum([ ( su2_decomposition_coefficients[i] * SU2_GENERATORS[i] ) for i in range(len(SU2_GENERATORS)) ]) #TODO numpy-ify the calculation

    assert is_hermitian(matrix), "matrix must be Hermitian"

    return matrix


def convert_su2_decomposition_coefficients_to_matrix_nusquids(su2_decomposition_coefficients) :
    '''
    This is doing the same thing as `convert_matrix_to_su2_decomposition_coefficients`,
    but using equations copied from SQuIDS (SU2ToMatrix.txt).

    Can be used as a cross-check
    '''

    assert is_real(su2_decomposition_coefficients), "coefficients must be real"

    matrix = np.array([
        [  su2_decomposition_coefficients[0] + su2_decomposition_coefficients[3]     ,  su2_decomposition_coefficients[1] - su2_decomposition_coefficients[2]*1j   ],
        [  su2_decomposition_coefficients[1] + su2_decomposition_coefficients[2]*1j  ,  su2_decomposition_coefficients[0] - su2_decomposition_coefficients[3]     ]
    ], dtype=np.complex128 )

    assert is_hermitian(matrix), "matrix must be Hermitian"

    return matrix


def convert_su3_decomposition_coefficients_to_matrix(su3_decomposition_coefficients) :
    '''
    Re-compose the linear combination of SU(2) generators representation of a 2x2 square matrix back into a 2X2 square matrix
    
    Read the documentation for `convert_su2_decomposition_coefficients_to_matrix` for more details
    '''

    assert is_real(su3_decomposition_coefficients), "coefficients must be real"

    # Here I directly use the SU(3) generators according to equation 15 in [2]
    matrix = sum([ ( su3_decomposition_coefficients[i] * SU3_GENERATORS[i] ) for i in range(len(SU3_GENERATORS)) ]) #TODO numpy-ify the calculation

    assert is_hermitian(matrix), "matrix must be Hermitian"

    return matrix


def convert_matrix_to_sun_decomposition_coefficients(matrix) :
    '''
    SU(N) wrapper for SU(2), SU(3), etc
    '''
    if matrix.shape[0] == 2 :
        return convert_matrix_to_su2_decomposition_coefficients(matrix)
    elif matrix.shape[0] == 3 :
        return convert_matrix_to_su3_decomposition_coefficients(matrix)
    else :
        raise Exception( "SU(%i) not currently supported" % (matrix.shape[0]) )


def convert_sun_decomposition_coefficients_to_matrix(sun_decomposition_coefficients) :
    '''
    SU(N) wrapper for SU(2), SU(3), etc
    '''
    if sun_decomposition_coefficients.size == 4 :
        return convert_su2_decomposition_coefficients_to_matrix(sun_decomposition_coefficients)
    elif sun_decomposition_coefficients.size == 9 :
        return convert_su3_decomposition_coefficients_to_matrix(sun_decomposition_coefficients)
    else :
        raise Exception( "SU(%i) not currently supported" % (np.sqrt(sun_decomposition_coefficients.size)-1) )


# def elementwise_product_of_su2_coefficients(A,B) :
#     '''
#     For two (Hermitian) matrices:
#       1) Convert both matrices to linear combinations of SU(2) generators
#       2) Calculate the element-wise product of the resulting coefficients
#       3) Convert the resulting coefficients back to a single matrix (using the generators again)
#     '''

#     # This is a hand-derived formula for the entire end-to-end process (e.g. sytesp 1+2+3)
#     # This is nice as can write it dow in a paper, but not so straightforward for SU(3)
#     AB_00 = ( ( A[0,0] * B[0,0] ) + ( A[1,1] * B[1,1] ) ) / 2.
#     AB_01 = ( A[0,1].real * B[0,1].real ) - 1.j * ( A[1,0].imag * B[1,0].imag )
#     AB_10 = ( A[0,1].real * B[0,1].real ) + 1.j * ( A[1,0].imag * B[1,0].imag )
#     AB_11 = ( ( A[0,0] * B[1,1] ) + ( A[1,1] * B[0,0] ) ) / 2.
#     return np.array([
#         [ AB_00, AB_01 ],
#         [ AB_10, AB_11 ],
#     ], dtype=np.complex128)


# def elementwise_product_of_su3_coefficients(A,B) :
#     #TODO
#     pass




#
# Test
#

if __name__ == "__main__" :

    np.set_printoptions(suppress=True,precision=6)


    #
    # SU(2) algebra
    #

    # Round trip test of SU(2) decomposition and recomposition 

    test_matrices = [
        np.array( [ [1,0], [0,-1] ], dtype=np.complex128 ),
        np.array( [ [1,0], [0,0] ], dtype=np.complex128 ),
        np.array( [ [1,1], [1,1] ], dtype=np.complex128 ),
    ]

    for m in test_matrices :
        assert is_hermitian(m)
        m_su2_cfts = convert_matrix_to_sun_decomposition_coefficients(m)
        m_again = convert_sun_decomposition_coefficients_to_matrix(m_su2_cfts)
        assert np.allclose( m, m_again, atol=1.e-6 ), "SU(2) test failed :\n%s\n!=\n%s"%(m, m_again)

    print("SU(2) test passed")


    #
    # SU(3) algebra
    #

    # Round trip test of SU(3) decomposition and recomposition 

    test_matrices = [
        np.array( [ [1,0,0], [0,1,0], [0,0,1] ], dtype=np.complex128 ),
        np.array( [ [1,1,1], [1,1,1], [1,1,1] ], dtype=np.complex128 ),
        np.array( [ [0,1,1], [1,0,1], [1,1,0] ], dtype=np.complex128 ),
        np.array( [ [-1,1,1], [1,1,1], [1,1,1] ], dtype=np.complex128 ),
        np.array( [ [0,0,-1], [0,0,0], [-1,0,0] ], dtype=np.complex128 ),
        np.array( [ [0,0,-1], [0,0,0], [-1,0,0] ], dtype=np.complex128 ),
        np.array( [ [-1,1,1], [1,-1,1], [1,1,1] ], dtype=np.complex128 ),
        np.array( [ [-1,1,1], [1,-1,1], [1,1,-1] ], dtype=np.complex128 ),
    ]

    for m in test_matrices :
        assert is_hermitian(m)
        m_su3_cfts = convert_matrix_to_sun_decomposition_coefficients(m)
        m_again = convert_sun_decomposition_coefficients_to_matrix(m_su3_cfts)
        assert np.allclose( m, m_again, atol=1.e-6 ), "SU(3) test failed :\n%s\n!=\n%s"%(m, m_again)

    print("SU(3) test passed")
