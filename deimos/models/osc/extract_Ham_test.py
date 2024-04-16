import numpy as np
import itertools
from scipy.spatial.transform import Rotation

from deimos.wrapper.osc_calculator import *


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

    # Check if the matrix is diagonalizable
    if not np.all(np.iscomplex(eigenvalues) | np.isreal(eigenvalues)):
        raise ValueError("Matrix is not diagonalizable.")

    return sorted_eigenvectors, eigenvalues


def pmns_angles_from_PMNS(matrix, deg=True):
    """
    Returns the mixing angles.
    """
    #assert is_unitary(matrix)

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

    # Check whether angles lie in first quadrant
    #if np.all(angles >= 0) and np.all(angles <= 90):
    #    pass
    #else:
    #    raise Exception("Angles are not in the first quadrant. The PMNS matrix might have been calculated using a wrong order. The calculated angels are ", angles)

    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)
    

def try_all_mixing_orders_and_signs(mixing_matrix, H2E, deg=True):
    """
    Returns the mixing angles that will diagonalize the Hamiltonian in ascending order of the diagonal elements.
    Has problems if two diagonal elements are equal.
    """
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
                    
                    result_array = np.dot(modified_matrix.conjugate().T, np.dot(H2E, modified_matrix))
                    # Set values close to zero to zero based on the tolerance
                    tolerance = 1e-9
                    result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

                    if np.diag(result_array)[0] <= np.diag(result_array)[1] and np.diag(result_array)[1] <= np.diag(result_array)[2]:
                        print("Angles in the first quadrant for sign combination", sign_combination, "and the following order of eigenvectors", order)
                        if deg:
                            return np.rad2deg(theta12), np.rad2deg(theta13), np.rad2deg(theta23)
                        else:
                            return theta12, theta13, theta23
                        
                    elif np.allclose(result_array, np.diag(np.diagonal(result_array))):
                        # Assert that result_array is diagonal
                        print("Matrix is diagonal, but not in ascending order of the diagonal elements.\n Diagonal matrix: ", result_array)
                        if deg:
                            return np.rad2deg(theta12), np.rad2deg(theta13), np.rad2deg(theta23)
                        else:
                            return theta12, theta13, theta23
                        
                    else:
                        raise Exception("No diagonalization of the Hamiltonian by calculated mixing angles.")
            else:
                raise Exception("Could not calculate mixing angles.")
                    

def diagonalize_H2E(H2E, deg=True):
    """
    Returns all mixing angles that diagonalize the matrix H2E.
    """
    # Calculate the mixing matrix
    mixing_matrix, eigenvalues = get_mixing_matrix(H2E)

    # Initialize a list to store the diagonalizing angles
    diagonalizing_angles = []
    diagonalized_matrices = []

    # Try all mixing orders and signs
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
                    result_array = np.dot(modified_matrix.conjugate().T, np.dot(H2E, modified_matrix))
                    # Set values close to zero to zero based on the tolerance
                    tolerance = 1e-9
                    result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

                    if np.allclose(result_array, np.diag(np.diagonal(result_array))):
                        # Assert that result_array is diagonal
                        diagonalizing_angles.append((theta12, theta13, theta23))
                        diagonalized_matrices.append(result_array)

    if deg:
        return [np.rad2deg(angles) for angles in diagonalizing_angles], diagonalized_matrices
    else:
        return diagonalizing_angles, diagonalized_matrices


energy_GeV = 1.
distance_km = 10000.

# Create calculator
calculator = OscCalculator(
    tool="nusquids",
    atmospheric=False,
    energy_nodes_GeV=energy_GeV,
    mass_splittings_eV2=[2e-5, 2e-3],
)

# Set some matter (to make sure there is a time-dependent component of the Hamiltonian)
calculator.set_matter(matter="constant", matter_density_g_per_cm3=0.0, electron_fraction=0.0)

# Calc osc probs (this inits the state in nuSQuIDS)
calculator.calc_osc_prob(
    energy_GeV=energy_GeV,
    distance_km=distance_km,
    initial_flavor=1,
    nubar=False,
)

print(dir(calculator.nusquids))
# Get Hamiltonian vs time (e.g. distance)
for x in np.linspace(0., distance_km, num=1) :
    H = calculator.nusquids.GetHamiltonianAtTime(x, 0, 0, True) # args: [x, E node, rho (nu vs nubar), flavor (True) or mass (False) basis]
    H = np.array(H[:,:,0] + H[:,:,1]*1j, dtype=np.complex128) # This properly builds the real and imaginary components
    print("\n x = %0.2f km \n%s" % (x, H*2*energy_GeV*1e9))

    # Calculate the mixing matrix
    mixing_matrix, eigenvalues = get_mixing_matrix(H, E_eV=energy_GeV*1e9)

    # Calculate the product of conjugate transpose of mixing_matrix, H, and mixing_matrix
    result_array = np.dot(mixing_matrix.conjugate().T, np.dot(H, mixing_matrix))*2*energy_GeV*1e9

    # Set values close to zero to zero based on the tolerance
    tolerance = 1e-9
    result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

    # Calculate mixing angles
    theta12, theta13, theta23 =try_all_mixing_orders_and_signs(mixing_matrix, H2E=H*energy_GeV*1e9, deg=True)

    print("Calculated mixing angles:")
    print(f"Theta12: {theta12} deg")
    print(f"Theta13: {theta13} deg")
    print(f"Theta23: {theta23} deg")

    # Calculate all mixing angles
    all_angles, mixing_matrices = diagonalize_H2E(H*energy_GeV*1e9, deg=True)
    print("All mixing angles:")
    print(all_angles)
    print("All mixing matrices:")
    print(mixing_matrices)
