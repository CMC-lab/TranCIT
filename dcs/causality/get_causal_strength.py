import numpy as np
from dcs.utils.preprocessing.singular import regularize_if_singular


def cs_nonzero_mean(X, morder, time_mode, diag_flag):
    """
    Causal Strength calculation with non-zero mean for bivariate signals.

    Parameters:
        X : np.ndarray
            Input data array of shape (nvar, nobs, ntrials), where nvar is the number of variables,
            nobs is the number of observations, and ntrials is the number of trials.
        morder : int
            Model order (number of lags).
        time_mode : str
            Time mode: 'inhomo' for time-inhomogeneous or 'homo' for homogeneous.
        diag_flag : int
            Flag to use diagonal covariance matrices (0 = full covariance, 1 = diagonal).

    Returns:
        causal_strength : np.ndarray
            Causal strength measures. Shape: (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
            Represents the strength of directional influence between variables.
        transfer_entropy : np.ndarray
            Transfer entropy measures. Shape: (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
            Quantifies information transfer between variables.
        granger_causality : np.ndarray
            Granger causality measures. Shape: (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
            Indicates predictive causality based on past values.
        coeff_t : np.ndarray
            Estimated coefficients. Shape: (nobs, nvar, nvar * morder + 1).
        transfer_entropy_residual_cov : np.ndarray
            Residual covariance for transfer entropy. Shape: (T, 2) where T = nobs - morder.
    """
    nvar, nobs, ntrials = X.shape
    
    r = morder
    r1 = morder + 1
    extended_X = np.zeros((nvar, r1, nobs + r1 - 1, ntrials))
    
    for k in range(r1):
        extended_X[:, k, k:k + nobs, :] = X
    
    current_X = extended_X[:, 0, r1 - 1:nobs, :]
    lagged_X = extended_X[:, 1:r+1, r1 - 1:nobs - 1, :]
    # T = nobs - morder
    T = current_X.shape[1] - 1
    cov_Xp, cov_Yp, C_XYp, C_YXp = compute_covariances(lagged_X, T, morder)
    
    transfer_entropy_residual_cov = np.zeros((T, 2))
    transfer_entropy = np.zeros((T, 2))
    causal_strength = np.zeros((T, 2))
    granger_causality = np.zeros((T, 2))
        
    coeff_t = np.full((nobs, nvar, nvar * morder + 1), np.nan)
    
    for t in range(T):
        coefficients, residual_cov = estimate_coefficients(current_X[:, t, :].T,
                                                        lagged_X[:, :morder, t, :].reshape(nvar * morder, ntrials).T,
                                                        ntrials)
        coeff_t[t, :, :] = coefficients
        A_square = coefficients[:, :-1].reshape(nvar, nvar, morder)
        b = A_square[0, 1, :]
        c = A_square[1, 0, :]
        sigy = residual_cov[0, 0]
        sigx = residual_cov[1, 1]
        
        cov_Yp_reg = regularize_if_singular(cov_Yp[t])
        cov_Xp_reg = regularize_if_singular(cov_Xp[t])
        
        if time_mode == 'inhomo':
            transfer_entropy_residual_cov[t, 1] = (
                sigy + b.T @ cov_Xp_reg @ b 
                - b.T @ C_XYp[t] @ np.linalg.inv(cov_Yp_reg) @ C_XYp[t].T @ b
            )
            transfer_entropy_residual_cov[t, 0] = (
                sigx + c.T @ cov_Yp_reg @ c 
                - c.T @ C_YXp[t] @ np.linalg.inv(cov_Xp_reg) @ C_YXp[t].T @ c
            )

            transfer_entropy[t, 1] = 0.5 * np.log(transfer_entropy_residual_cov[t, 1] / sigy)
            transfer_entropy[t, 0] = 0.5 * np.log(transfer_entropy_residual_cov[t, 0] / sigx)
            
            if not diag_flag:
                causal_strength[t, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp_reg @ b) / sigy)
                causal_strength[t, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp_reg @ c) / sigx)
            else:
                causal_strength[t, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp_reg)) @ b) / sigy)
                causal_strength[t, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp_reg)) @ c) / sigx)
                
            
            lagged_X_p = lagged_X[1, :, t, :].T
            lagged_Y_p = lagged_X[0, :, t, :].T
            current_X_t = current_X[1, t, :]
            current_Y_t = current_X[0, t, :]
            _, sigx_r = estimate_coefficients(current_X_t, lagged_X_p, ntrials)
            _, sigy_r = estimate_coefficients(current_Y_t, lagged_Y_p, ntrials)
            granger_causality[t, 1] = np.log(sigy_r / sigy)
            granger_causality[t, 0] = np.log(sigx_r / sigx)
        
    if time_mode == 'homo':
        cov_Xp = np.mean(cov_Xp, axis=0)
        cov_Yp = np.mean(cov_Yp, axis=0)
        Sig_Xp = np.mean(Sig_Xp, axis=0)
        Sig_Yp = np.mean(Sig_Yp, axis=0)
        C_XYp = np.mean(C_XYp, axis=0)
        C_YXp = np.mean(C_YXp, axis=0)
        S_Xp = np.mean(S_Xp, axis=0)
        S_Yp = np.mean(S_Yp, axis=0)
        
        transfer_entropy = np.zeros((1, 2))
        causal_strength = np.zeros((1, 2))
        granger_causality = np.zeros((1, 2))
        
        cov_Yp_reg = regularize_if_singular(cov_Yp)
        cov_Xp_reg = regularize_if_singular(cov_Xp)
        
        transfer_entropy[0, 1] = 0.5 * np.log((sigy + b.T @ Sig_Xp @ b - b.T @ C_XYp @ np.linalg.inv(Sig_Yp) @ C_XYp.T @ b) / sigy)
        transfer_entropy[0, 0] = 0.5 * np.log((sigx + c.T @ Sig_Yp @ c - c.T @ C_YXp @ np.linalg.inv(Sig_Xp) @ C_YXp.T @ c) / sigx)
        
        if not diag_flag:
            causal_strength[0, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp_reg @ b) / sigy)
            causal_strength[0, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp_reg @ c) / sigx)
        else:
            causal_strength[0, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp_reg)) @ b) / sigy)
            causal_strength[0, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp_reg)) @ c) / sigx)
        
        granger_causality[0, 1] = np.log(sigy_r / sigy)
        granger_causality[0, 0] = np.log(sigx_r / sigx)
        
    if time_mode == 'inhomo':
        nan_block = np.full((morder, 2), np.nan)
        
        granger_causality = np.vstack([nan_block, granger_causality])
        transfer_entropy = np.vstack([nan_block, transfer_entropy])
        causal_strength = np.vstack([nan_block, causal_strength])

    return causal_strength, transfer_entropy, granger_causality, coeff_t, transfer_entropy_residual_cov

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
