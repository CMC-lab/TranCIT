import numpy as np
from .core import regularize_if_singular
def compute_covariances(lagged_data, T, morder):
    """
    Compute covariance matrices for lagged data.
    
    Parameters:
    - lagged_data: Lagged data array (nvar x morder x T x ntrials)
    - T: Number of time steps
    - morder: Model order (number of lags)
    
    Returns:
    - cov_Xp: Covariance of X past
    - cov_Yp: Covariance of Y past
    - C_XYp: Cross-covariance X past to Y past
    - C_YXp: Cross-covariance Y past to X past
    """
    cov_Xp = np.zeros((T, morder, morder))
    cov_Yp = np.zeros((T, morder, morder))
    C_XYp = np.zeros((T, morder, morder))
    C_YXp = np.zeros((T, morder, morder))
    
    for t in range(T):
        X_p = lagged_data[1, :, t, :].T
        Y_p = lagged_data[0, :, t, :].T
        cov_Xp[t] = np.cov(X_p.T)
        cov_Yp[t] = np.cov(Y_p.T)
        C_XYp[t] = np.cov(X_p.T, Y_p.T)[:morder, morder:]
        C_YXp[t] = np.cov(Y_p.T, X_p.T)[:morder, morder:]
    
    return cov_Xp, cov_Yp, C_XYp, C_YXp


def estimate_coefficients(current_data, lagged_data, ntrials):
    """
    Compute regression coefficients and residual covariance.
    
    Parameters:
    - current_data: Current time step data (nvar x ntrials)
    - lagged_data: Lagged data (nvar * morder x ntrials)
    - ntrials: Number of trials
    
    Returns:
    - Coeff: Regression coefficients
    - SIG: Residual covariance matrix
    """
    lagged_data_ones = np.hstack([lagged_data, np.ones((ntrials, 1))])
    cross_cov_current = np.dot(current_data.T, current_data) / ntrials
    cross_cov_between = np.dot(current_data.T, lagged_data_ones) / ntrials
    auto_cov_lagged = np.dot(lagged_data_ones.T, lagged_data_ones) / ntrials
    auto_cov_lagged = regularize_if_singular(auto_cov_lagged)
    coeff = np.linalg.solve(auto_cov_lagged, cross_cov_between.T).T
    residual_cov = cross_cov_current - np.dot(coeff, np.dot(auto_cov_lagged, coeff.T))
    
    return coeff, residual_cov
