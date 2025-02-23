import numpy as np
from .signals import morlet

def gen_ensemble_nonstat_innomean(A, SIG, ntrials, L_event, center, amp, dim, L_perturb):
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
        X[:, :, n], Imp[:, :, n] = genvar_nonstat(A, SIG, morder, nvar, L_event, amp, dim, L_perturb, center)
    
    return X, Imp

def genvar_nonstat(A, SIG, morder, nvar, L_event, amp, dim, L_perturb, center):
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
    # Imp[dim, start_idx:end_idx] = Imp_shape[:end_idx - start_idx]
    Imp[dim, start_idx:end_idx] = Imp_shape
    
    for t in range(morder , L_event):
        # X_lag = np.flip(X[:, t - morder:t], axis=1).reshape(nvar * morder, 1)
        X_lag = np.flip(X[:, t - morder:t]).flatten()
        
        X[:, t] += A @ X_lag
        # X[:, t] += np.dot(A, X_lag).flatten()
        X[:, t] += Imp[:, t - morder]
    
    X = X[:, morder:]    
    return X, Imp
