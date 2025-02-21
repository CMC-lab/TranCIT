import numpy as np

def coefficient_estimation_compare(X, morder, momax, time_mode, lag_mode, epsilon=1e-8):
    """
    Estimate coefficients of a VAR model and compute residual covariance matrix with regularization.
    
    Args:
    X: 3D array of shape (nvar, nobs, ntrials) representing the time series data.
    morder: Model order for the VAR process.
    momax: Maximum model order for comparison.
    time_mode: Time mode ('inhomo' or 'homo').
    lag_mode: Lag mode ('infocrit' or 'var').
    epsilon: Regularization term to handle singular matrices (default: 1e-8).
    
    Returns:
    A: Coefficient matrix of the VAR model.
    SIG: Residual covariance matrix.
    logL: Log-likelihood.
    sum_detHess: Sum of log determinants of the Hessian.
    """
    
    nvar, nobs, ntrials = X.shape
    
    r = morder
    
    if lag_mode == 'infocrit':
        r1 = momax + 1
    elif lag_mode == 'var':
        r1 = morder + 1
    
    XX = np.zeros((nvar, r1, nobs + r1 - 1, ntrials))
    
    for k in range(r1):
        XX[:, k, k:k + nobs, :] = X
    
    # X0 = XX[:, 0, r1 - 1:nobs + r1 - 1]
    X0 = XX[:, 0, r1 - 1:nobs, :]
    # XL = XX[:, 1:morder + 1, r1 - 1:nobs + r1 - 1]
    XL = XX[:, 1 : r + 1, r1 - 1 :nobs, :]
    T = X0.shape[1]
    
    A = np.zeros((T, nvar, nvar * morder))
    SIG = np.zeros((T, nvar, nvar))
    DSIG = np.zeros(T)
    log_detHess = np.zeros(T)
    detHess = np.zeros(T)
    
    for t in range(T):
        if nvar != 1:
            Ct_0 = np.dot(X0[:, t, :], X0[:, t, :].T) / ntrials
            YX_lag = np.vstack([
                                XL[:, :morder, t, :].reshape(nvar * morder, ntrials),
                                np.ones((1, ntrials))
                                ])
            
            Ct_j = np.dot(X0[:, t, :], YX_lag.T) / ntrials
            Ct_1r = np.dot(YX_lag, YX_lag.T) / ntrials
            
            # Ct_1r += epsilon * np.eye(Ct_1r.shape[0])
        
            try:
                Coeff = np.linalg.solve(Ct_1r, Ct_j.T).T
            except np.linalg.LinAlgError:
                print(f"Singular matrix encountered at t={t}, applying regularization.")
                Coeff = np.linalg.pinv(Ct_1r) @ Ct_j.T
        else:
            raise print("X should be bivariate signals!!!!")
        
        if time_mode == 'inhomo':
            A[t, :, :] = Coeff[:, :-1]
            
            if nvar != 1:
                SIG[t, :, :] = Ct_0 - np.dot(Coeff, np.dot(Ct_1r, Coeff.T))
                DSIG[t] = np.prod(np.diag(SIG[t, :, :]))
            else:
                SIG[t, :, :] = Ct_0 - np.dot(A[t, :, :].T, np.dot(Ct_1r, A[t, :, :]))
                DSIG[t] = np.prod(np.diag(SIG[t, :, :]))
            
            C = Ct_1r[:-1, :-1] * ntrials
            det_C = np.linalg.det(C)
            detHess[t] = det_C**nvar * (1 / DSIG[t])**(nvar * morder)
            
            C_0 = Ct_1r[:-1, :-1]
            log_detHess[t] = morder * nvar**2 * np.log(ntrials) + nvar * np.log(np.linalg.det(C_0)) - nvar * morder * np.log(DSIG[t])
    
    if time_mode == 'inhomo':
        DSIG_clamped = np.where(DSIG < epsilon, epsilon, DSIG)
        logL = -0.5 * T * nvar * np.log(2 * np.pi) - 0.5 * np.sum(np.log(DSIG_clamped)) - 0.5 * T * nvar
        sum_detHess = np.sum(log_detHess)
    
    if time_mode == 'homo':
        C_0 = np.mean(Ct_0, axis=0)
        C_jcr = np.mean(Ct_j, axis=0)
        C_1rcr = np.mean(Ct_1r, axis=0)
        
        A = np.dot(C_jcr.reshape(nvar, nvar * morder), np.linalg.inv(C_1rcr))
        
        SIG = C_0 - np.dot(A, np.dot(C_1rcr, A.T))
        
        sumsign = np.sum(np.sign(SIG))
        DSIG = np.prod(np.diag(SIG))
        
        C = C_1rcr * T * ntrials
        detHess = (np.linalg.det(C)**nvar) * ((1 / DSIG)**(nvar * morder))
        
        logL = -0.5 * T * nvar * np.log(2 * np.pi) - 0.5 * T * np.log(DSIG) - 0.5 * T * nvar
        
        sum_detHess = nvar * np.log(np.linalg.det(C)) + (nvar * morder) * np.log(1 / DSIG)
    
    return A, SIG, logL, sum_detHess
