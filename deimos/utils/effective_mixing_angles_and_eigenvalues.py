import numpy as np
from scipy.spatial.transform import Rotation
import itertools

#
# Methods
#

def pmns_angles_from_PMNS(matrix, deg=True):
    """
    Returns the mixing angles.
    """
    # Convert the matrix to a rotation object
    rotation = Rotation.from_matrix(matrix)
    
    # Use canonical rotation order
    order = 'zyx'
    angles = rotation.as_euler(order, degrees=True)

    # Correct for opposite sign of theta23 and theta12 compared to physics convention
    '''
    Code to check signs of angles
    r_z = R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r_y = R.from_matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    r_x = R.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r.as_euler('xyz', degrees=True)
    '''
    angles[0] = -angles[0]
    angles[2] = -angles[2]
        
    # Extract the angles of rotation around z, y, and x axes
    theta12, theta13, theta23 = angles 

    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)
    

def get_mixing_matrix(matrix, E_eV=1):
    """
    Calculates the mixing matrix and eigenvalues of a given Hamiltonian matrix.
    """
    # Rescale matrix to account for energy dependence
    matrix = matrix*2*E_eV

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    

    return sorted_eigenvectors, sorted_eigenvalues


def diagonalize_HE(HE, deg=True):
    """
    Output: mixing angles, mixing matrices and eigenvalues of the Hamiltonian matrix
    Note: Outputs all mixing angles that diagonalize the Hamiltonian matrix. Might not be unique.
    H = \Delta m^2 / 2 E
    """
    # Calculate the mixing matrix
    mixing_matrix, eigenvalues = get_mixing_matrix(HE)

    # Initialize a list to store the diagonalizing angles
    diagonalizing_angles = []
    diagonalized_matrices = []

    # Try all mixing orders
    n = mixing_matrix.shape[0]
    signs = [-1, 1]
    column_orders = list(itertools.permutations(range(n)))

    for order in column_orders:
        for sign_combination in itertools.product(signs, repeat=n):
            modified_matrix = mixing_matrix[:, order] * np.array(sign_combination)
            angles = pmns_angles_from_PMNS(matrix=modified_matrix, deg=False)

            if angles is not None:
                theta12, theta13, theta23 = angles
                
                if 0-0.01 <= theta12 <= np.pi/2+0.01 and 0-0.01 <= theta13 <= np.pi/2+0.01 and 0-0.01 <= theta23 <= np.pi/2+0.01:
                    result_array = np.dot(modified_matrix.conjugate().T, np.dot(HE, modified_matrix))
                    # Set values close to zero to zero based on the tolerance
                    tolerance = 1e-9
                    result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

                    # Assert that result_array is diagonal
                    if np.allclose(result_array, np.diag(np.diagonal(result_array))):
                        diagonalizing_angles.append((theta12, theta13, theta23))
                        diagonalized_matrices.append(result_array)

        
        if len(diagonalizing_angles) > 1:
            # Sort diagonalizing angles based on negative angles
            diagonalizing_angles.sort(key=lambda angles: any(angle < 0 for angle in angles))
            # Sort diagonalizing angles based on non-zero angles
            diagonalizing_angles.sort(key=lambda angles: sum(angle != 0 for angle in angles), reverse=True)

    if deg:
        return [np.rad2deg(angles) for angles in diagonalizing_angles], diagonalized_matrices, eigenvalues
    else:
        return diagonalizing_angles, diagonalized_matrices, eigenvalues


#
# Methods only used for testing the code
#
    
def generate_neutrino_hamiltonian(mass_squared_diff, mixing_angles, E_eV):
    """
    Generate a 3x3 neutrino flavor Hamiltonian matrix based on mass squared differences and mixing angles.

    Parameters:
    - mass_squared_diff (tuple): Mass squared differences in eV^2.
    - mixing_angles (triple): Vacuum mixing angles in radians.

    Returns:
    - numpy.ndarray: 3x3 neutrino flavor Hamiltonian matrix.
    """
    # Check if input lists/tuples have correct length
    if len(mass_squared_diff) != 2 or len(mixing_angles) != 3:
        raise ValueError("Incorrect input dimensions. Mass squared differences and mixing angles should have length 2.")

    # Check if mixing angles are in the first quadrant
    if not np.all((0 <= np.array(mixing_angles)) & (np.array(mixing_angles) <= np.pi/2)):
        raise ValueError("Mixing angles should be in the first quadrant.")
    
    # Unpack mass squared differences and mixing angles
    delta_m21_sq, delta_m31_sq = mass_squared_diff
    #theta12, theta13, theta23 = mixing_angles

    # Construct the neutrino flavor Hamiltonian matrix
    H = np.zeros((3, 3))

    # Diagonal elements (normal mass hierarchy)
    H[1, 1] = delta_m21_sq
    H[2, 2] = delta_m31_sq
    
    # PMNS matrix
    U_PMNS = get_pmns_matrix(mixing_angles, dcp=0.)

    #flavor state matrix
    H_f = 1/(2*E_eV) * np.dot(U_PMNS, np.dot(H, U_PMNS.conjugate().T))

    return H_f


def get_pmns_matrix(theta, dcp=0.) :
    """ Get the PMNS matrix (rotation from mass to flavor basis)"""

    if len(theta) == 1 :
        assert (dcp is None) or np.isclose(dcp, 0.)
        # This is just the standard unitary rotation matrix in 2D
        pmns = np.array( [ [np.cos(theta[0]),np.sin(theta[0])], [-np.sin(theta[0]),np.cos(theta[0])] ], dtype=np.complex128 )
    elif len(theta) == 3 :
        # Check if mixing angles are in the first quadrant
        #if not np.all((0 <= np.array(theta)) & (np.array(theta) <= np.pi/2)):
        #    raise ValueError("Mixing angles should be in the first quadrant.")
        
        # Using definition from https://en.wikipedia.org/wiki/Neutrino_oscillation
        pmns = np.array( [
            [   np.cos(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[1])*np.exp(1.j*dcp)  ], #Janni
            [  -np.sin(theta[0])*np.cos(theta[2]) -np.cos(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[0])*np.cos(theta[2]) -np.sin(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.sin(theta[2])*np.cos(theta[1])  ], 
            [  np.sin(theta[0])*np.sin(theta[2]) -np.cos(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    -np.cos(theta[0])*np.sin(theta[2]) -np.sin(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[2])*np.cos(theta[1])  ],
        ], dtype=np.complex128 )
    else :
        raise Exception("Only 2x2 or 3x3 PMNS matrix supported")

    return pmns


if __name__ == "__main__" :

    # Set random values for mass squared differences and mixing angles
    for i in np.random.uniform(0, np.pi/2, 1):
        j = np.random.uniform(0, np.pi/2)
        k = np.random.uniform(0, np.pi/2)
        m2 = np.random.uniform(0, 1)
        m1 = np.random.uniform(0, m2)
        nu_mass_squared_diff = [m1, m2]
        mixing_angles = [i, j, k]

        # Calculate the neutrino Hamiltonian
        Ham = generate_neutrino_hamiltonian(nu_mass_squared_diff, mixing_angles, 1/2)

        #
        # Check that get_mixing_matrix method works
        #


        # Calculate the mixing matrix
        mixing_matrix, eigenvalues = get_mixing_matrix(Ham, E_eV=1/2)

        # Calculate the product of conjugate transpose of mixing_matrix, H, and mixing_matrix
        result_array = np.dot(mixing_matrix.conjugate().T, np.dot(Ham, mixing_matrix))

        # Set values close to zero to zero based on the tolerance
        tolerance = 1e-9
        result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

        # Check if result_array is diagonal
        if np.allclose(result_array, np.diag(np.diagonal(result_array))):
            print("Matrix is diagonal.")
        else:
            print("Matrix is not diagonal.")
        
        # Check if diagonal elements are close to the mass squared differences
        if np.allclose(np.diag(result_array)[1:], nu_mass_squared_diff):
            print("Diagonal elements are correct.")
        else:
            print("Diagonal elements are incorrect.")


        #
        # Check that diagonalize_HE method works
        #

        # Calculate mixing angles
        angles, _, _  =diagonalize_HE(HE=Ham, deg=False)

        theta12, theta13, theta23 = angles[0]

        # Check if calculated mixing angles are correct
        if np.allclose([theta12, theta13, theta23], mixing_angles):
            print("Angles are correct.")
        else:
            print("Difference between calculated and input mixing angles:")
            print(f"\Delta Theta12: {theta12-mixing_angles[0]} rad")
            print(f"\Delta Theta13: {theta13-mixing_angles[1]} rad")
            print(f"\Delta Theta23: {theta23-mixing_angles[2]} rad")
