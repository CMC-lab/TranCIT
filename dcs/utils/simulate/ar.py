import numpy as np

def simul_AR_event_btsp(simobj, Yt_event, Yt_stats, Et):
    """
    Simulate autoregressive (AR) events with bootstrapping.
    
    Parameters:
    simobj : dict
        A dictionary containing simulation parameters:
        - nvar : int, number of variables
        - morder : int, model order
        - L : int, length of each trial
        - Ntrials : int, number of trials
    Yt_event : np.ndarray
        Original event data of shape (nvar * (morder + 1), L, trials).
    Yt_stats : dict
        Dictionary containing OLS coefficients under `OLS.At` of shape (L, nvar, nvar * morder).
    Et : np.ndarray
        Residuals of shape (nvar, L, trials).
        
    Returns:
    Yt_event_btsp : np.ndarray
        Bootstrapped event data of shape (nvar * (morder + 1), L, trials).
    """
    nvar, morder, L, Ntrials = simobj['nvar'], simobj['morder'], simobj['L'], simobj['Ntrials']
    Yt_event_btsp = np.full((nvar * (morder + 1), L, Ntrials), np.nan)
    
    for n in range(Ntrials):
        y = np.zeros((nvar, L))
        
        # Fill y with bootstrapped residuals
        for t in range(L):
            rand_trial = np.random.randint(0, Yt_event.shape[2])
            y[:, t] = Et[:, t, rand_trial]  # Shape: (nvar,)
        
        rand_trial = np.random.randint(0, Yt_event.shape[2])
        rand_Yt0 = Yt_event[:, 0, rand_trial]  # Shape: (nvar*(morder+1),)
        lagged_vars = rand_Yt0[2:].reshape(2, simobj['morder'])[::-1]  # Shape: (2, morder), reversed
        y = np.hstack((lagged_vars, y))  # Shape: (nvar, morder + L
        
        for ktime in range(morder, y.shape[1]):
            coeff = Yt_stats['OLS']['At'][ktime - morder, :, :].reshape(nvar, nvar, morder)
            for kdelay in range(1, morder + 1):
                y[:, ktime] += np.dot(coeff[:, :, kdelay - 1], y[:, ktime - kdelay])
        
        for delay in range(simobj['morder']):
            start_idx = nvar * delay
            end_idx = nvar * (delay + 1)
            Yt_event_btsp[start_idx:end_idx, :, n] = y[:, morder - delay - 1:-(delay + 1)]
    
    return Yt_event_btsp


def simul_AR_event(simobj, Yt_stats):
    """
    Simulates AR events with non-stationary innovations using the specified simulation object and statistics.

    Parameters:
    - simobj (dict): Simulation settings with fields:
        - nvar (int): Number of variables.
        - morder (int): Model order.
        - L (int): Length of the event.
        - Ntrials (int): Number of trials.
    - Yt_stats (dict): Statistics with fields:
        - OLS.At (numpy.ndarray): Autoregressive coefficients.
        - OLS.Sigma_Et (numpy.ndarray): Covariance matrices for errors.
        - OLS.bt (numpy.ndarray): Mean term for innovations.
        - mean (numpy.ndarray): Mean state.
        - Sigma (numpy.ndarray): Covariance matrix of the state.

    Returns:
    - Yt_event (numpy.ndarray): Simulated AR events.
    """
    nvar, morder, L, Ntrials = simobj['nvar'], simobj['morder'], simobj['L'], simobj['Ntrials']
    Yt_event = np.full((nvar * (morder + 1), L, Ntrials), np.nan)

    for n in range(Ntrials):
        y = np.zeros((nvar, L))

        # Generate innovations with covariance sigma at each time point
        for t in range(L):
            sigma = np.round(Yt_stats['OLS']['Sigma_Et'][t], 2)
            if np.linalg.cond(sigma) > 1 / np.finfo(sigma.dtype).eps:
                sigma = 0.5 * (Yt_stats['OLS']['Sigma_Et'][t - 1] + Yt_stats['OLS']['Sigma_Et'][t + 1])
            y[:, t] = np.random.multivariate_normal(Yt_stats['OLS']['bt'][:, t], sigma)

        rand_Yt0 = np.random.multivariate_normal(Yt_stats['mean'][:, 0], Yt_stats['Sigma'][0, :, :])
        y = np.hstack([np.flipud(rand_Yt0[nvar:].reshape(nvar, morder)), y])

        for ktime in range(morder, y.shape[1]):
            coeff = Yt_stats['OLS']['At'][ktime - morder].reshape(nvar, nvar, morder)
            for kdelay in range(morder):
                y[:, ktime] += coeff[:, :, kdelay] @ y[:, ktime - kdelay - 1]

        Yt_event[:nvar, :, n] = y[:, morder:]
        for delay in range(1, morder + 1):
            Yt_event[nvar * delay:nvar * (delay + 1), :, n] = y[:, morder - delay:L - delay + morder]

    return Yt_event

def simul_AR_kaidi_nonstat_innomean(A, SIG, innomean, morder):
    """
    Simulate a non-stationary autoregressive (AR) process with innovations.
    
    Args:
    A: Coefficient matrix of the AR process.
    SIG: Covariance matrix of the innovations.
    innomean: Innovations (input mean).
    morder: Number of lags in the AR model.
    
    Returns:
    y: Simulated AR process (nvar x L).
    """
    
    nvar, L = innomean.shape
    y = np.random.multivariate_normal(mean=np.zeros(2), cov=SIG, size=L).T
    y += innomean

    for t in range(morder+1, L):
        
        temp = np.flip(y[:, t - morder:t], axis=1).reshape(nvar * morder, 1)
        y[:, t] += (A @ temp).squeeze()
    
    return y
