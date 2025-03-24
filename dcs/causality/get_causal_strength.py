import numpy as np
from dcs.utils.preprocessing.singular import regularize_if_singular


def cs_nonzero_mean(X, morder, time_mode, diag_flag):
    """
    Causal Strength calculation with non-zero mean for bivariate signals.
    
    Parameters:
    X : np.ndarray
        The input data array with shape (nvar, nobs, ntrials).
    morder : int
        The model order.
    time_mode : str
        The mode for time: 'inhomo' for time-inhomogeneous or 'homo' for homogeneous.
    diag_flag : int
        A flag indicating whether to use diagonal covariance matrices.
    
    Returns:
    causal_strength : np.ndarray
        Causal strength measures.
    transfer_entropy : np.ndarray
        Transfer entropy measures.
    granger_causality : np.ndarray
        Granger causality measures.
    coeff_t : np.ndarray
        The estimated coefficients.
    transfer_entropy_residual_cov : np.ndarray
        The residual covariance for transfer entropy.
    """
    nvar, nobs, ntrials = X.shape
    
    r = morder
    r1 = morder + 1
    XX = np.zeros((nvar, r1, nobs + r1 - 1, ntrials))
    
    for k in range(r1):
        XX[:, k, k:k + nobs, :] = X
    
    X0 = XX[:, 0, r1 - 1:nobs, :]
    XL = XX[:, 1:r+1, r1 - 1:nobs - 1, :]
    # T = nobs - morder
    T = X0.shape[1] - 1
    
    transfer_entropy_residual_cov = np.zeros((T, 2))
    transfer_entropy = np.zeros((T, 2))
    causal_strength = np.zeros((T, 2))
    granger_causality = np.zeros((T, 2))
    
    cov_Xp = np.zeros((T, morder, morder))
    cov_Yp = np.zeros((T, morder, morder))
    C_XYp = np.zeros((T, morder, morder))
    C_YXp = np.zeros((T, morder, morder))
        
    coeff_t = np.full((nobs, nvar, nvar * morder + 1), np.nan)
    
    for t in range(T):
        Ct_0 = np.dot(X0[:, t, :], X0[:, t, :].T) / ntrials
        YX_lag = np.vstack([XL[:, :morder, t, :].reshape(nvar * morder, ntrials), np.ones((1, ntrials))])
        Ct_j = np.dot(X0[:, t, :], YX_lag.T) / ntrials
        Ct_1r = np.dot(YX_lag, YX_lag.T) / ntrials
        Ct_1r = regularize_if_singular(Ct_1r)
        Coeff = np.linalg.solve(Ct_1r, Ct_j.T).T
        SIG = Ct_0 - np.dot(Coeff, np.dot(Ct_1r, Coeff.T))
        
        coeff_t[t, :, :] = Coeff
        
        A_square = Coeff[:, :-1].reshape(nvar, nvar, morder)
        
        a, b = A_square[0, 0, :], A_square[0, 1, :]
        sigy = SIG[0, 0]
        
        c, d = A_square[1, 0, :], A_square[1, 1, :]
        sigx = SIG[1, 1]

        X_p = XL[1, :, t, :].T
        Y_p = XL[0, :, t, :].T
        
        X_t = X0[1, t, :]
        Y_t = X0[0, t, :]
        
        X_lag = np.hstack([X_p, np.ones((ntrials, 1))])
        Ct_0_x = np.dot(X_t.T, X_t) / ntrials
        Ct_j_x = np.dot(X_t.T, X_lag) / ntrials
        Ct_1r_x = np.dot(X_lag.T, X_lag) / ntrials
        Ct_1r_x = regularize_if_singular(Ct_1r_x)
        Coeff_x = np.linalg.solve(Ct_1r_x, Ct_j_x.T).T
        sigx_r = Ct_0_x - np.dot(Coeff_x, np.dot(Ct_1r_x, Coeff_x.T))
        
        Y_lag = np.hstack([Y_p, np.ones((ntrials, 1))])
        Ct_0_y = np.dot(Y_t.T, Y_t) / ntrials
        Ct_j_y = np.dot(Y_t.T, Y_lag) / ntrials
        Ct_1r_y = np.dot(Y_lag.T, Y_lag) / ntrials
        Ct_1r_y = regularize_if_singular(Ct_1r_y)
        Coeff_y = np.linalg.solve(Ct_1r_y, Ct_j_y.T).T
        sigy_r = Ct_0_y - np.dot(Coeff_y, np.dot(Ct_1r_y, Coeff_y.T))
        
        cov_Xp[t, :, :] = np.cov(X_p.T)
        cov_Yp[t, :, :] = np.cov(Y_p.T)
        C_XYp[t, :, :] = np.cov(X_p.T, Y_p.T)[:nvar, nvar:]
        C_YXp[t, :, :] = np.cov(Y_p.T, X_p.T)[:nvar, nvar:]
        
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
                
            granger_causality[t, 1] = np.log(sigy_r / sigy)
            granger_causality[t, 0] = np.log(sigx_r / sigx)
        
    if time_mode == 'homo':
        A_square = A_square.reshape(nvar, nvar, morder)
        
        a = A_square[0, 0, :]
        b = A_square[0, 1, :]
        sigy = SIG[0, 0]
        
        c = A_square[1, 0, :]
        d = A_square[1, 1, :]
        sigx = SIG[1, 1]
        
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
