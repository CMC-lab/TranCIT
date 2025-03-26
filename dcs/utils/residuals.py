import numpy as np
from typing import Dict, Tuple

def estimate_residuals(Yt_stats: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L, nvar, temp = Yt_stats['OLS']['At'].shape

    bt = np.full((nvar, L), np.nan)
    Sigma_Et = np.full((L, nvar, nvar), np.nan)
    sigma_Et = np.full((L, 1), np.nan)

    for t in range(L):
        Sigma_Xt = np.squeeze(Yt_stats['Sigma'][t, :nvar, :nvar])
        Sigma_Xp = np.squeeze(Yt_stats['Sigma'][t, nvar:, nvar:])
        Sigma_XtXp = np.reshape(np.squeeze(Yt_stats['Sigma'][t, :nvar, nvar:]), (nvar, temp))
        coeff = np.reshape(np.squeeze(Yt_stats['OLS']['At'][t, :, :]), (nvar, temp))

        bt[:, t] = Yt_stats['mean'][:nvar, t] - np.dot(coeff, Yt_stats['mean'][nvar:, t])
        Sigma_Et[t, :, :] = (Sigma_Xt - 
                             np.dot(Sigma_XtXp, coeff.T) - 
                             np.dot(coeff, Sigma_XtXp.T) + 
                             np.dot(np.dot(coeff, Sigma_Xp), coeff.T))
        
        sigma_Et[t] = np.trace(np.squeeze(Sigma_Et[t, :, :]))

    # Ensure that no negative values exist (if needed)
    # Sigma_Et[Sigma_Et < 0] = 0
    # sigma_Et[sigma_Et < 0] = 0

    return bt, Sigma_Et, sigma_Et

def get_residuals(Yt_event: np.ndarray, Yt_stats: Dict) -> np.ndarray:
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
