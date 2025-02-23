import numpy as np
from .residuals import estimate_residuals


def get_Yt_stats_cond(Yt_event, mo):
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


def get_Yt(y, loc, mo, tau, L_start, L_extract):
    nvar = y.shape[0]
    Yt = np.full((nvar * (mo + 1), L_extract, len(loc)), np.nan)
    
    idx1 = np.arange(nvar * (mo + 1))
    idx2 = np.tile(np.arange(nvar), mo + 1)
    delay = np.tile(np.arange(0, mo + 1) * tau, (nvar, 1)).flatten()

    for n in range(len(idx1)):
        Yt[idx1[n], :, :] = extract_events(y[idx2[n], :], loc - delay[n], L_start, L_extract)
    
    return Yt


def extract_events(A, cumP, L_start, L):
    A_event = np.full((L, len(cumP)), np.nan)
    
    for i in range(len(cumP)):
        start_idx = int(np.round(cumP[i] - L_start + 1))
        end_idx = int(np.round(cumP[i] + L - L_start + 1))
        idx = np.arange(start_idx, end_idx).astype(int)

        if np.any(idx < 0) or np.any(idx >= len(A)):
            raise IndexError(f"Index out of bounds: {idx} for array of length {len(A)}")
        
        A_event[:, i] = A[idx]
    
    return A_event


# def get_Yt_stats(Yt_event, mo):
#     nvar = Yt_event.shape[0] // (mo + 1)
#     Yt_stats_cond = {}

#     Yt_stats_cond['mean'] = np.mean(Yt_event, axis=2)
#     Yt_stats_cond['Ntrials'] = Yt_event.shape[2]

#     Yt_stats_cond['Sigma'] = np.zeros((Yt_event.shape[1], nvar * (mo + 1), nvar * (mo + 1)))
#     Yt_stats_cond['OLS'] = {'At': np.zeros((Yt_event.shape[1], nvar, nvar * mo))}

#     for t in range(Yt_event.shape[1]):
#         temp = Yt_event[:, t, :] - Yt_stats_cond['mean'][:, t][:, np.newaxis]
#         Yt_stats_cond['Sigma'][t, :, :] = np.dot(temp, temp.T) / Yt_event.shape[2]
#         Yt_stats_cond['OLS']['At'][t, :, :] = np.reshape(
#             np.linalg.solve(
#                 Yt_stats_cond['Sigma'][t, nvar:nvar * (mo + 1), nvar:nvar * (mo + 1)],
#                 Yt_stats_cond['Sigma'][t, :nvar, nvar:nvar * (mo + 1)]
#             ).T,
#             (nvar, nvar * mo)
#         )

#     Yt_stats_cond['OLS']['bt'], Yt_stats_cond['OLS']['Sigma_Et'], Yt_stats_cond['OLS']['sigma_Et'] = estimate_residuals(Yt_stats_cond)

#     return Yt_stats_cond

def get_Yt_stats(Yt_event, mo):
    nvar = Yt_event.shape[0] // (mo + 1)
    Yt_stats_cond = {}

    Yt_stats_cond['mean'] = np.mean(Yt_event, axis=2)
    Yt_stats_cond['Ntrials'] = Yt_event.shape[2]

    Yt_stats_cond['Sigma'] = np.zeros((Yt_event.shape[1], nvar * (mo + 1), nvar * (mo + 1)))
    Yt_stats_cond['OLS'] = {'At': np.zeros((Yt_event.shape[1], nvar, nvar * mo))}

    for t in range(Yt_event.shape[1]):
        temp = Yt_event[:, t, :] - Yt_stats_cond['mean'][:, t][:, np.newaxis]
        Yt_stats_cond['Sigma'][t, :, :] = np.dot(temp, temp.T) / Yt_event.shape[2]
        
        Sigma_Y = Yt_stats_cond['Sigma'][t, nvar:nvar * (mo + 1), nvar:nvar * (mo + 1)]
        Sigma_X = Yt_stats_cond['Sigma'][t, :nvar, nvar:nvar * (mo + 1)]

        # Ensure that Sigma_Y is square and that Sigma_X has compatible dimensions
        if Sigma_Y.shape[0] != Sigma_Y.shape[1] or Sigma_Y.shape[0] != Sigma_X.shape[1]:
            raise ValueError(f"Shape mismatch between Sigma_Y {Sigma_Y.shape} and Sigma_X {Sigma_X.shape}")

        # Solve the linear system
        Yt_stats_cond['OLS']['At'][t, :, :] = np.reshape(
            np.linalg.solve(Sigma_Y, Sigma_X.T),
            (nvar, nvar * mo)
        )

    Yt_stats_cond['OLS']['bt'], Yt_stats_cond['OLS']['Sigma_Et'], Yt_stats_cond['OLS']['sigma_Et'] = estimate_residuals(Yt_stats_cond)

    return Yt_stats_cond
