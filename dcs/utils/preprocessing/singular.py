import numpy as np

def regularize_if_singular(matrix, epsilon=1e-6, threshold=1e-6):
    """
    Check if a matrix is singular and regularize it by adding epsilon to the diagonal if needed.
    
    Args:
        matrix (ndarray): Square matrix to check and potentially regularize.
        epsilon (float, optional): Small value to add to the diagonal if singular. Default is 1e-6.
        threshold (float, optional): Determinant threshold below which the matrix is considered singular. Default is 1e-10.
    
    Returns:
        ndarray: The original matrix if non-singular, or the regularized matrix if singular.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")
    # cond_threshold = 1e10
    # cond_number = np.linalg.cond(matrix)
    
    # if cond_number > cond_threshold:
    #     epsilon = 1e-6
    #     print(f"Warning: Near-singular matrix at t={t}, cond={cond_number}, adding epsilon={epsilon}")
    #     regularized_matrix = matrix + epsilon * np.eye(matrix.shape[0])
    #     return regularized_matrix
    # else:
    #     return matrix
    
    det = np.linalg.det(matrix)
    if abs(det) < threshold:
        print(f"Warning: Singular matrix detected (det={det}), adding epsilon={epsilon}")
        regularized_matrix = matrix + epsilon * np.eye(matrix.shape[0])
        return regularized_matrix
    else:
        return matrix
