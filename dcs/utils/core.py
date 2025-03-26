import numpy as np

from .preprocess import regularize_if_singular
from .residuals import estimate_residuals
from typing import Dict


def extract_event_windows(A: np.ndarray, cumP: np.ndarray, L_start: int, L: int) -> np.ndarray:
    A_event = np.full((L, len(cumP)), np.nan)
    
    for i in range(len(cumP)):
        start_idx = int(np.round(cumP[i] - L_start))
        end_idx = int(np.round(cumP[i] + L - L_start))
        idx = np.arange(start_idx, end_idx).astype(int)

        if np.any(idx < 0) or np.any(idx >= len(A)):
            raise IndexError(f"Index out of bounds: {idx} for array of length {len(A)}")
        
        A_event[:, i] = A[idx]
    
    return A_event

def compute_conditional_event_statistics(Yt_event: np.ndarray, mo: int) -> Dict:
    """
    Compute conditional statistics of the VAR time series events.
    
    Args:
    Yt_event: The VAR time series events (nvar*(mo+1) x time points x trials).
    mo: Model order.
    
    Returns:
    Yt_stats_cond: Dictionary containing the conditional statistics, including mean and covariance.
    """
    
    nvar = Yt_event.shape[0] // (mo + 1)
    Yt_stats_cond = {}
    
    Yt_stats_cond = {
        "mean": np.mean(Yt_event, axis=2),
        "Sigma": np.zeros((Yt_event.shape[1], nvar * (mo + 1), nvar * (mo + 1))),
        "OLS": {
            "At": np.zeros((Yt_event.shape[1], nvar, nvar * mo))
        }
    }

    for t in range(Yt_event.shape[1]):
        temp = Yt_event[:, t, :] - Yt_stats_cond['mean'][:, t, np.newaxis]
        Yt_stats_cond['Sigma'][t, :, :] = np.dot(temp, temp.T) / Yt_event.shape[2]
        
        # Yt_stats_cond['OLS']['At'][t, :, :] = np.reshape(Yt_stats_cond['Sigma'][t, :nvar, nvar:], (nvar, nvar * mo)) / Yt_stats_cond['Sigma'][t, nvar:, nvar:]
        Sigma_sub_matrix = Yt_stats_cond["Sigma"][t, :nvar, nvar:]
        Sigma_end = Yt_stats_cond["Sigma"][t, nvar:, nvar:]
        
        epsilon = 1e-4
        if np.linalg.det(Sigma_end) == 0:
            Sigma_end += np.eye(Sigma_end.shape[0]) * epsilon
        
        Yt_stats_cond["OLS"]["At"][t, :, :] = np.dot(Sigma_sub_matrix, np.linalg.inv(Sigma_end))
    
    Yt_stats_cond['OLS']['bt'], Yt_stats_cond['OLS']['Sigma_Et'], Yt_stats_cond['OLS']['sigma_Et'] = estimate_residuals(Yt_stats_cond)

    return Yt_stats_cond


def extract_event_snapshots(y: np.ndarray, loc: np.ndarray, mo: int, tau: int, L_start: int, L_extract: int) -> np.ndarray:
    nvar = y.shape[0]
    Yt = np.full((nvar * (mo + 1), L_extract, len(loc)), np.nan)
    
    idx1 = np.arange(nvar * (mo + 1))
    idx2 = np.tile(np.arange(nvar), mo + 1)
    delay = np.tile(np.arange(0, mo + 1) * tau, (nvar, 1)).flatten()

    for n in range(len(idx1)):
        Yt[idx1[n], :, :] = extract_event_windows(y[idx2[n], :], loc - delay[n], L_start, L_extract)
    
    return Yt


def compute_event_statistics(Yt_event: np.ndarray, mo: int) -> Dict:
    nvar = Yt_event.shape[0] // (mo + 1)
    nobs = Yt_event.shape[1]
    Ntrials = Yt_event.shape[2]
    
    Yt_stats_cond = {}
    Yt_stats_cond['mean'] = np.mean(Yt_event, axis=2)
    Yt_stats_cond['Ntrials'] = Ntrials
    Yt_stats_cond['Sigma'] = np.zeros((nobs, nvar * (mo + 1), nvar * (mo + 1)))
    Yt_stats_cond['OLS'] = {'At': np.zeros((nobs, nvar, nvar * mo))}

    for t in range(nobs):
        temp = Yt_event[:, t, :] - Yt_stats_cond['mean'][:, t][:, np.newaxis]
        Yt_stats_cond['Sigma'][t, :, :] = np.dot(temp, temp.T) / Ntrials
    
        Sigma_12 = Yt_stats_cond['Sigma'][t, :nvar, nvar:]  # Shape: (nvar, nvar*mo)
        Sigma_22 = Yt_stats_cond['Sigma'][t, nvar:, nvar:]  # Shape: (nvar*mo, nvar*mo)
        
        Sigma_22 = regularize_if_singular(Sigma_22)
        Yt_stats_cond['OLS']['At'][t, :, :] = Sigma_12 @ np.linalg.inv(Sigma_22)

    bt, Sigma_Et, sigma_Et = estimate_residuals(Yt_stats_cond)
    Yt_stats_cond['OLS']['bt'] = bt
    Yt_stats_cond['OLS']['Sigma_Et'] = Sigma_Et
    Yt_stats_cond['OLS']['sigma_Et'] = sigma_Et

    return Yt_stats_cond
