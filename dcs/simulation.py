from typing import Dict, Tuple

import numpy as np


def simulate_ar_event_bootstrap(simobj: Dict, Yt_event: np.ndarray, Yt_stats: Dict, Et: np.ndarray) -> np.ndarray:
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


def simulate_ar_event(simobj: Dict, Yt_stats: Dict) -> np.ndarray:
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

def simulate_ar_nonstat_innomean(A: np.ndarray, SIG: np.ndarray, innomean: np.ndarray, morder: int) -> np.ndarray:
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


def generate_signals(T: int, Ntrial: int, h: float, gamma1: float, gamma2: float, Omega1: float, Omega2: float, apply_morlet: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate bivariate coupled oscillator signals based on a specified model.

    Simulates two coupled second-order linear differential equations discretized
    using a time step h, with added noise terms. Allows for optional non-stationarity
    in the noise applied to the first signal (x) via a Morlet wavelet shape.

    Parameters
    ----------
    T : int
        Total number of time points to simulate (including initial points).
    Ntrial : int
        Number of trials (realizations) to generate.
    h : float
        Time step for discretization.
    gamma1 : float
        Damping coefficient for the first oscillator (x).
    gamma2 : float
        Damping coefficient for the second oscillator (y).
    Omega1 : float
        Natural frequency for the first oscillator (x).
    Omega2 : float
        Natural frequency for the second oscillator (y).
    apply_morlet : bool, optional
        If True, applies a Morlet wavelet shape to modulate the noise variance (`ns_x`)
        for the first signal, introducing non-stationarity. Defaults to False, using
        constant noise variance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - X : Generated signals, shape (2, T - 500, Ntrial). Contains x and y signals after discarding the first 500 points.
        - ns_x : Noise variance profile for the x signal, shape (T + 1,).
        - ns_y : Noise variance profile for the y signal, shape (T + 1,).
    """
    X = np.zeros((2, T - 500, Ntrial))
    
    if apply_morlet == True:
        ns_x = 0.02 * np.concatenate([
            np.ones(650), 
            np.ones(201) - morlet(-0.29, 0.29, 201), 
            np.ones(150)
        ])
    else:
        ns_x = 0.02 * np.ones(T + 1)
        
    ns_y = 0.005 * np.ones(T + 1)
    
    for N in range(Ntrial):
        x = np.random.rand(2)
        y = np.random.rand(2)
        
        c2 = 0
        c1 = 0.098

        for t in range(1, T - 1):
            x = np.append(x, (2 - gamma1*h) * x[-1] + (-1 + gamma1*h - h**2 * Omega1**2) * x[-2] + h**2 * ns_x[t] * np.random.randn() + h**2 * c2 * y[-2])
            y = np.append(y, (2 - gamma2*h) * y[-1] + (-1 + gamma2*h - h**2 * Omega2**2) * y[-2] + h**2 * ns_y[t] * np.random.randn() + h**2 * c1 * x[-2])

        u = np.array([x[500:], y[500:]])
        X[:, :, N] = u
    
    return X, ns_x, ns_y

def morlet(start: float, end: float, num_points: int) -> np.ndarray:
    """
    Generate a Morlet wavelet.
    
    Args:
    start: Start frequency for the Morlet wavelet.
    end: End frequency for the Morlet wavelet.
    num_points: Length of the perturbation.
    
    Returns:
    Morlet wavelet.
    """
    t = np.linspace(start, end, num_points)
    w0 = 5
    sigma = (end - start) / (2 * np.pi)
    # sigma = end / (2 * np.pi)  # Standard deviation for Gaussian
    wavelet = (1 / np.sqrt(sigma)) * np.exp(1j * w0 * t) * np.exp(-t ** 2 / (2 * (sigma ** 2)))
    # wavelet = np.exp(1j * start * t) * np.exp(-t ** 2 / (2 * (end ** 2))) # For perturbation
    return np.real(wavelet)

def generate_ensemble_nonstat_innomean(A: np.ndarray, SIG: np.ndarray, ntrials: int, L_event: int, center: int, amp: float, dim: int, L_perturb: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an ensemble of non-stationary VAR processes.
    
    Args:
    A: Coefficient matrix of the VAR process.
    SIG: Covariance matrix of the innovations.
    ntrials: Number of trials/realizations.
    L_event: Length of each event (time series).
    center: Center point for perturbation.
    amp: Amplitude of perturbation.
    dim: Dimension of variables the input goes into.
    L_perturb: Length of perturbation.
    
    Returns:
    X: Generated VAR process ensemble (nvar x L_event x ntrials).
    Imp: The impulse matrix for the generated processes.
    """
    
    nvar, temp = A.shape
    morder = temp // nvar
    
    X = np.empty((nvar, L_event, ntrials))
    Imp = np.empty((nvar, L_event, ntrials))
    
    for n in range(ntrials):  # For each realization
        X[:, :, n], Imp[:, :, n] = generate_var_nonstat(A, SIG, morder, nvar, L_event, amp, dim, L_perturb, center)
    
    return X, Imp

def generate_var_nonstat(A: np.ndarray, SIG: np.ndarray, morder: int, nvar: int, L_event: int, amp: float, dim: int, L_perturb: int, center: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a non-stationary VAR process.
    
    Args:
    A: Coefficient matrix of the VAR process.
    SIG: Covariance matrix of the innovations.
    morder: Number of lags in the VAR model.
    nvar: Number of variables.
    L_event: Length of the event (time series).
    amp: Amplitude of the perturbation.
    dim: Dimension of variables the input goes into.
    L_perturb: Length of perturbation.
    center: Center point for perturbation.
    
    Returns:
    X: Generated VAR process (nvar x L_event).
    Imp: Impulse matrix (2 x L_event).
    """
    
    # Initialise to Gaussian white noise
    X = SIG @ np.random.randn(nvar, L_event + morder) # "SIG" is actually Cholesky matrix
    if L_perturb == 1:
        Imp_shape = amp * 1
    else:
        Imp_shape = amp * morlet(-4, 4, L_perturb) # 101 point long morlet wave
    
    start_idx = center - (L_perturb // 2)
    end_idx = center + (L_perturb // 2)
    
    Imp = np.zeros((2, L_event))
    Imp[dim, start_idx:end_idx] = Imp_shape
    
    for t in range(morder + 1, L_event):
        X_lag = np.flip(X[:, t - morder:t ], axis=1).reshape(nvar * morder, 1)
        X[:, t] = X[:, t] + (A @ X_lag).flatten()
        X[:, t] = X[:, t] + Imp[:, t - morder]
    
    X = X[:, morder:]
    return X, Imp

