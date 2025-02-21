import numpy as np
from estimate_coef import coefficient_estimation_compare

def model_order_selection_compare(X, momax, time_mode):
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
            _, _, logL[mo-1], sum_detHess[mo-1] = coefficient_estimation_compare(X, mo, momax, time_mode, 'infocrit')
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
