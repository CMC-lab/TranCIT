import numpy as np

def get_residuals(Yt_event, Yt_stats):
    """
    Calculate residuals for each time step based on the model coefficients.
    
    Parameters:
    Yt_event : np.ndarray
        The event matrix with shape (nvar * (morder + 1), L, ntrials).
    Yt_stats : dict
        A dictionary containing model statistics, specifically `OLS.At` 
        with shape (L, nvar, nvar * morder).

    Returns:
    Et : np.ndarray
        The residuals with shape (nvar, L, ntrials).
    """
    L, nvar, _ = Yt_stats['OLS']['At'].shape
    ntrials = Yt_event.shape[2]
    Et = np.full((nvar, L, ntrials), np.nan)
    
    for t in range(L):
        Xt = Yt_event[:nvar, t, :]  # Shape: (nvar, ntrials)
        Xp = Yt_event[nvar:, t, :]   # Shape: (nvar * morder, ntrials)
        coeff = Yt_stats['OLS']['At'][t]  # Shape: (nvar, nvar * morder)
        
        Et[:, t, :] = Xt - np.dot(coeff, Xp)

    return Et
