from typing import Dict, Tuple

import numpy as np
from dcs.utils.core.getting_Yt import get_Yt_stats

# from dcs.causality.detecting_analysis_pipeline import snapshot_detect_analysis_pipeline

def multi_trial_BIC(Yt_events_momax: np.ndarray, BICParser: Dict) -> Dict:
    """
    Calculate BIC (Bayesian Information Criterion) for multiple trial event data and model orders.
    
    Args:
    Yt_events_momax : np.ndarray
        The time series data with shape (nvar * (mo + 1), nobs, ntrials)
    BICParser : object
        The object containing the BIC parameters (including model order, `Params.BIC.momax`)
    
    Returns:
    BICoutputs : dict
        Dictionary containing BIC values and associated metrics
    """
    
    momax = BICParser["Params"]["BIC"]["momax"]
    temp, nobs, ntrials = Yt_events_momax.shape
    nvar = temp // (momax + 1)

    BICoutputs = {
        'bic': np.full((momax, 4), np.nan),
        'pt_bic': np.full((momax, 4), np.nan),
        'logL': np.full(momax, np.nan),
        'sum_detHess': np.full(momax, np.nan),
        'mobic': None
    }

    for mo in range(1, momax + 1):
        print(f'Start calculation for model order: {mo}')
        X = Yt_events_momax[:nvar * (mo + 1), :, :]

        logL, sum_detHess = BIC_compare(X, mo, BICParser)
        BICoutputs['logL'][mo-1] = logL
        BICoutputs['sum_detHess'][mo-1] = sum_detHess
        
        BICoutputs['pt_bic'][mo-1, 0] = 0.5 * nobs * mo * nvar * nvar * np.log(ntrials)
        BICoutputs['pt_bic'][mo-1, 1] = 0.5 * sum_detHess
        BICoutputs['pt_bic'][mo-1, 2] = 0.5 * nobs * mo * nvar * nvar * np.log(ntrials * nobs)
        BICoutputs['pt_bic'][mo-1, 3] = 0.5 * mo * nvar * nvar * np.log(ntrials * nobs)

        BICoutputs['bic'][mo-1, 0] = -BICoutputs['logL'][mo-1] * ntrials + BICoutputs['pt_bic'][mo-1, 0]
        BICoutputs['bic'][mo-1, 1] = -BICoutputs['logL'][mo-1] * ntrials + BICoutputs['pt_bic'][mo-1, 1]
        BICoutputs['bic'][mo-1, 2] = -BICoutputs['logL'][mo-1] * ntrials + BICoutputs['pt_bic'][mo-1, 2]
        BICoutputs['bic'][mo-1, 3] = -BICoutputs['logL'][mo-1] * ntrials + BICoutputs['pt_bic'][mo-1, 3]
    
    # bic_min = np.nanmin(BICoutputs['bic'], axis=0)
    mobic_index = np.nanargmin(BICoutputs['bic'], axis=0) + 1
    BICoutputs['mobic'] = mobic_index

    return BICoutputs


def BIC_compare(Yt_events, morder, BICParser):
    """
    Compare Bayesian Information Criterion (BIC) for biased and debiased models.
    
    Args:
    Yt_events : np.ndarray
        The event data with shape (nvar * (morder + 1), nobs, ntrials).
    morder : int
        The model order.
    BICParser : object
        An object containing BIC parameters.
    
    Returns:
    logL : float
        Log-likelihood.
    sum_detHess : float
        Sum of the log determinant of Hessian.
    """
    temp, nobs, ntrials = Yt_events.shape
    nvar = temp // (morder + 1)

    if BICParser['Params']['BIC']['mode'] == 'biased':
        Yt_stats = get_Yt_stats(Yt_events, morder)

    # elif BICParser['Params']['BIC']['mode'] == 'debiased':
    #     Params = BICParser['Params']
    #     Params['Options']['BIC'] = 0
    #     Params['Options']['save_flag'] = 0
    #     Params['BIC']['morder'] = morder
    #     SnapAnalyOutput = snapshot_detect_analysis_pipeline(BICParser['OriSignal'], 
    #                                                         BICParser['DetSignal'],
    #                                                         Params)
    #     Yt_stats = SnapAnalyOutput['Yt_stats_debiased']

    log_detHess = np.zeros(nobs)
    DSIG = np.zeros(nobs)

    for t in range(nobs):
        C_0 = np.squeeze(Yt_stats['Sigma'][t, nvar:, nvar:])
        
        if BICParser['EstimMode'] == 'OLS':
            DSIG[t] = np.prod(np.diag(np.squeeze(Yt_stats['OLS']['Sigma_Et'][t, :, :])))
            
        elif BICParser['EstimMode'] == 'RLS':
            DSIG[t] = np.prod(np.diag(np.squeeze(Yt_stats['RLS']['Sigma_Et'][t, :, :])))
            
        log_detHess[t] = (morder * nvar**2 * np.log(ntrials) + 
                          nvar * np.log(np.linalg.det(C_0)) - 
                          nvar * morder * np.log(DSIG[t]))

    logL = (-0.5 * nobs * nvar * np.log(2 * np.pi) - 
            0.5 * np.sum(np.log(DSIG)) -
            0.5 * nobs * nvar)
    sum_detHess = np.sum(log_detHess)

    return logL, sum_detHess


def select_model_order(X: np.ndarray, momax: int, time_mode: str) -> Tuple[np.ndarray, ...]:
    """
    Perform model order selection using the Bayesian Information Criterion (BIC) and calculate log-likelihood.
    
    Args:
    X: 3D array of shape (nvar, nobs, ntrials) representing the time series data.
    momax: Maximum model order for selection.
    time_mode: String indicating 'inhomo' (inhomogeneous) or 'homo' (homogeneous) time mode.
    
    Returns:
    bic: Array of BIC scores for each model order.
    mobic: The selected model order that minimizes the BIC.
    logL: Log-likelihood for each model order.
    pt_bic: Penalty term used for calculating BIC.
    """
    
    nvar, nobs, ntrials = X.shape

    bic = np.full((momax, 2), np.nan)
    pt_bic = np.full((momax, 2), np.nan)
    logL = np.full(momax, np.nan)
    sum_detHess = np.full(momax, np.nan)

    for mo in range(1, momax + 1):
        T = len(range(mo + 1, nobs))
        
        print(f"Processing model order: {mo}")

        try:
            _, _, logL[mo-1], sum_detHess[mo-1] = estimate_var_coefficients(X, mo, momax, time_mode, 'infocrit')
        except np.linalg.LinAlgError:
            print(f"Singular matrix encountered at model order {mo}, skipping.")
            continue

        print(f"logL[{mo-1}] = {logL[mo-1]}, sum_detHess[{mo-1}] = {sum_detHess[mo-1]}")

        pt_bic[mo-1, 1] = sum_detHess[mo-1]

        if time_mode == 'inhomo':
            pt_bic[mo-1, 0] = T * mo * nvar * nvar * np.log(ntrials)
            bic[mo-1, 0] = - logL[mo-1] * ntrials + pt_bic[mo-1, 0]
            bic[mo-1, 1] = - logL[mo-1] * ntrials + sum_detHess[mo-1]
        elif time_mode == 'homo':
            pt_bic[mo-1, 0] = mo * nvar * nvar * np.log(T * ntrials)
            bic[mo-1, 0] = - logL[mo-1] * ntrials + pt_bic[mo-1, 0]
            bic[mo-1, 1] = - logL[mo-1] * ntrials + sum_detHess[mo-1]


        print(f"BIC[{mo-1}] = {bic[mo-1]}")

    print("Final BIC array:")
    print(bic)

    if np.isnan(bic).all():
        raise ValueError("All BIC values are NaN. Check data or model order settings. Ensure the input data X is valid, and try reducing momax.")

    mobic = np.nanargmin(bic, axis=0) + 1

    return bic, mobic, logL, pt_bic

def estimate_var_coefficients(X: np.ndarray, morder: int, momax: int, time_mode: str, lag_mode: str, epsilon: float = 1e-8) -> Tuple[np.ndarray, ...]:
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
