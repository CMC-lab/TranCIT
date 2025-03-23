import numpy as np
from dcs.utils.core.getting_Yt import get_Yt_stats
# from dcs.causality.detecting_analysis_pipeline import snapshot_detect_analysis_pipeline

def multi_trial_BIC(Yt_events_momax, BICParser):
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


