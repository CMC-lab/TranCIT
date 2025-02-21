import numpy as np

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
    Cs : np.ndarray
        Causal strength measures.
    TE : np.ndarray
        Transfer entropy measures.
    GC : np.ndarray
        Granger causality measures.
    coeff_t : np.ndarray
        The estimated coefficients.
    TE_residual_cov : np.ndarray
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
        
        epsilon = 1e-6
        Ct_1r += epsilon * np.eye(Ct_1r.shape[0])
        
        try:
            Coeff = np.linalg.solve(Ct_1r, Ct_j.T).T
        except np.linalg.LinAlgError:
            print(f"Singular matrix encountered at t={t}, applying regularization.")
            Coeff = np.linalg.pinv(Ct_1r) @ Ct_j.T
        
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
        
        try:
            Coeff_x = np.linalg.solve(Ct_1r_x, Ct_j_x.T).T
        except np.linalg.LinAlgError as e:
            print(f"Singular matrix error at time {t} in reduced X model: {e}")
            continue
        
        sigx_r = Ct_0_x - np.dot(Coeff_x, np.dot(Ct_1r_x, Coeff_x.T))
        
        Y_lag = np.hstack([Y_p, np.ones((ntrials, 1))])
        Ct_0_y = np.dot(Y_t.T, Y_t) / ntrials
        Ct_j_y = np.dot(Y_t.T, Y_lag) / ntrials
        Ct_1r_y = np.dot(Y_lag.T, Y_lag) / ntrials
        
        try:
            Coeff_y = np.linalg.solve(Ct_1r_y, Ct_j_y.T).T
        except np.linalg.LinAlgError as e:
            print(f"Singular matrix error at time {t} in reduced Y model: {e}")
            continue
        
        sigy_r = Ct_0_y - np.dot(Coeff_y, np.dot(Ct_1r_y, Coeff_y.T))
        
        # # Covariance calculations
        # cov_Xp[t, :, :] = np.cov(X_p.T)
        # cov_Yp[t, :, :] = np.cov(Y_p.T)
        # C_XYp[t, :, :] = np.cov(X_p.T, Y_p.T)[:nvar, nvar:]
        # C_YXp[t, :, :] = np.cov(Y_p.T, X_p.T)[:nvar, nvar:]

        if morder == 1:
            cov_Xp[t, :, :] = np.dot((X_p - np.mean(X_p, axis=0)), (X_p - np.mean(X_p, axis=0)).T) / ntrials
            cov_Yp[t, :, :] = np.dot((Y_p - np.mean(Y_p, axis=0)), (Y_p - np.mean(Y_p, axis=0)).T) / ntrials
            C_XYp[t, :, :] = np.dot((X_p - np.mean(X_p, axis=0)), (Y_p - np.mean(Y_p, axis=0)).T) / ntrials
            C_YXp[t, :, :] = np.dot((Y_p - np.mean(Y_p, axis=0)), (X_p - np.mean(X_p, axis=0)).T) / ntrials
        else:
            cov_Xp[t, :, :] = np.dot((X_p - np.mean(X_p, axis=0)).T, (X_p - np.mean(X_p, axis=0))) / ntrials
            cov_Yp[t, :, :] = np.dot((Y_p - np.mean(Y_p, axis=0)).T, (Y_p - np.mean(Y_p, axis=0))) / ntrials
            C_XYp[t, :, :] = np.dot((X_p - np.mean(X_p, axis=0)).T, (Y_p - np.mean(Y_p, axis=0))) / ntrials
            C_YXp[t, :, :] = np.dot((Y_p - np.mean(Y_p, axis=0)).T, (X_p - np.mean(X_p, axis=0))) / ntrials
        
        cov_Yp_reg = cov_Yp[t] + epsilon * np.eye(cov_Yp[t].shape[0])
        cov_Xp_reg = cov_Xp[t] + epsilon * np.eye(cov_Xp[t].shape[0])
        
        if time_mode == 'inhomo':
            TE_residual_cov = np.zeros((T, 2))
            TE = np.zeros((T, 2))
            Cs = np.zeros((T, 2))
            GC = np.zeros((T, 2))

            for t in range(T):
                
                # epsilon = 1e-6
                # cov_Yp += epsilon * np.eye(cov_Yp.shape)
                
                # TE_residual_cov[t, 1] = (
                #     sigy + b.T @ cov_Xp[t] @ b 
                #     - b.T @ C_XYp[t] @ np.linalg.inv(cov_Yp[t]) @ C_XYp[t].T @ b
                # )
                # TE_residual_cov[t, 0] = (
                #     sigx + c.T @ cov_Yp[t] @ c 
                #     - c.T @ C_YXp[t] @ np.linalg.inv(cov_Xp[t]) @ C_YXp[t].T @ c
                # )
                
                TE_residual_cov[t, 1] = (
                    sigy + b.T @ cov_Xp_reg @ b 
                    - b.T @ C_XYp[t] @ np.linalg.inv(cov_Yp_reg) @ C_XYp[t].T @ b
                )
                TE_residual_cov[t, 0] = (
                    sigx + c.T @ cov_Yp_reg @ c 
                    - c.T @ C_YXp[t] @ np.linalg.inv(cov_Xp_reg) @ C_YXp[t].T @ c
                )
                
                epsilon = 1e-8
                cov_Yp[t] += epsilon * np.eye(cov_Yp[t].shape[0])
                cov_Xp[t] += epsilon * np.eye(cov_Xp[t].shape[0])

                # TE[t, 1] = 0.5 * np.log(TE_residual_cov[t, 1] / sigy)
                # TE[t, 0] = 0.5 * np.log(TE_residual_cov[t, 0] / sigx)
            
                TE[t, 1] = 0.5 * np.log(sigy + b.T @ cov_Xp[t] @ b - b.T @ C_XYp[t] @ np.linalg.inv(cov_Yp[t]) @ C_XYp[t].T @ b / sigy)
                TE[t, 0] = 0.5 * np.log(sigx + c.T @ cov_Yp[t] @ c - c.T @ C_YXp[t] @ np.linalg.inv(cov_Xp[t]) @ C_YXp[t].T @ c / sigx)
                
                # if not diag_flag:
                #     Cs[t, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp[t] @ b) / sigy)
                #     Cs[t, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp[t] @ c) / sigx)
                # else:
                #     Cs[t, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp[t])) @ b) / sigy)
                #     Cs[t, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp[t])) @ c) / sigx)
                
                if not diag_flag:
                    Cs[t, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp_reg @ b) / sigy)
                    Cs[t, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp_reg @ c) / sigx)
                else:
                    Cs[t, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp_reg)) @ b) / sigy)
                    Cs[t, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp_reg)) @ c) / sigx)
                    
                GC[t, 1] = np.log(sigy_r / sigy)
                GC[t, 0] = np.log(sigx_r / sigx)
        
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
        
        TE = np.zeros((1, 2))
        
        epsilon = 1e-8
        cov_Yp += epsilon * np.eye(cov_Yp.shape[0])
        
        TE[0, 1] = 0.5 * np.log(sigy + b.T @ Sig_Xp @ b - b.T @ C_XYp @ np.linalg.inv(Sig_Yp) @ C_XYp.T @ b / sigy)
        TE[0, 0] = 0.5 * np.log(sigx + c.T @ Sig_Yp @ c - c.T @ C_YXp @ np.linalg.inv(Sig_Xp) @ C_YXp.T @ c / sigx)
        
        Cs = np.zeros((1, 2))
        if not diag_flag:
            Cs[0, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp @ b) / sigy)
            Cs[0, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp @ c) / sigx)
        else:
            Cs[0, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp)) @ b) / sigy)
            Cs[0, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp)) @ c) / sigx)
        
        GC = np.zeros((1, 2))
        GC[0, 1] = np.log(sigy_r / sigy)
        GC[0, 0] = np.log(sigx_r / sigx)
        
        
    if time_mode == 'inhomo':
        # Creating arrays of NaNs with the desired shape
        nan_block = np.full((morder, 2), np.nan)
        
        # Concatenating along the first axis (equivalent to stacking rows)
        GC = np.vstack([nan_block, GC])
        TE = np.vstack([nan_block, TE])
        Cs = np.vstack([nan_block, Cs])

    return abs(Cs), abs(TE), abs(GC)
