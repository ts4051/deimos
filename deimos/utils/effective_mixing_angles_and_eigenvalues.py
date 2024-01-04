import numpy as np
from scipy.spatial.transform import Rotation
import itertools
from itertools import permutations

def get_mixing_matrix(matrix, E_eV=1):
    # Rescale matrix to account for energy dependence
    matrix = matrix*2*E_eV

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Check if the matrix is diagonalizable
    if not np.all(np.iscomplex(eigenvalues) | np.isreal(eigenvalues)):
        raise ValueError("Matrix is not diagonalizable.")

    return sorted_eigenvectors, eigenvalues


def pmns_angles_from_PMNS(matrix, deg=True):
    """
    Returns the mixing angles.
    """

    # Convert the matrix to a rotation object
    rotation = Rotation.from_matrix(matrix)
    
    # Try canonical rotation order
    order = 'zyx'
    angles = rotation.as_euler(order, degrees=True)

    # Correct for wrong sign of theta23 and theta12
    angles[0] = -angles[0]
    angles[2] = -angles[2]
        
    # Extract the angles of rotation around x, y, and z axes
    theta12, theta13, theta23 = angles 

    # Return angles in degrees or radians
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)


def try_all_mixing_orders_and_signs(Hamiltonian2E, deg=True):
    """
    Returns the mixing angles that will diagonalize the matrix in ascending order of the diagonal elements.
    Has problems if two diagonal elements are equal.
    """
    mixing_matrix = get_mixing_matrix(Hamiltonian2E, E_eV=1)

    n = mixing_matrix.shape[0]
    # All possible sign combinations
    signs = [-1, 1]
    # All possible orders of the eigenvectors
    column_orders = list(itertools.permutations(range(n)))

    # Try all combinations of signs and orders
    for order in column_orders:
        for sign_combination in itertools.product(signs, repeat=n):
            modified_matrix = mixing_matrix[:, order] * np.array(sign_combination)
            theta12, theta13, theta23 = pmns_angles_from_PMNS(matrix=modified_matrix, deg=False)
            
            # Check whether angles lie in first quadrant
            if 0 <= theta12 <= np.pi/2 and 0 <= theta13 <= np.pi/2 and 0 <= theta23 <= np.pi/2:
                
                result_array = np.dot(modified_matrix.conjugate().T, np.dot(Hamiltonian2E, modified_matrix))
                # Set values close to zero to zero based on the tolerance
                tolerance = 1e-9
                result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

                # Check whether the diagonal elements are in ascending order
                if np.diag(result_array)[0] <= np.diag(result_array)[1] and np.diag(result_array)[1] <= np.diag(result_array)[2]:
                    
                    # Print the sign and the order of eigenvectors ()
                    print("Angles in the first quadrant for sign combination", sign_combination, "and the following order of eigenvectors", order)
                    
                    # Return angles in degrees or radians
                    if deg:
                        return np.rad2deg(theta12), np.rad2deg(theta13), np.rad2deg(theta23)
                    else:
                        return theta12, theta13, theta23