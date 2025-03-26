import numpy as np
from typing import Tuple, Dict
from utils.core import regularize_if_singular
from utils.helpers import compute_covariances, estimate_coefficients

def compute_causal_strength_nonzero_mean(X: np.ndarray, morder: int, time_mode: str, diag_flag: int) -> Tuple[np.ndarray, ...]:
    """
    Causal Strength calculation with non-zero mean for bivariate signals.

    Parameters:
        X : np.ndarray
            Input data array of shape (nvar, nobs, ntrials), where nvar is the number of variables,
            nobs is the number of observations, and ntrials is the number of trials.
        morder : int
            Model order (number of lags).
        time_mode : str
            Time mode: 'inhomo' for time-inhomogeneous or 'homo' for homogeneous.
        diag_flag : int
            Flag to use diagonal covariance matrices (0 = full covariance, 1 = diagonal).

    Returns:
        causal_strength : np.ndarray
            Causal strength measures. Shape: (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
            Represents the strength of directional influence between variables.
        transfer_entropy : np.ndarray
            Transfer entropy measures. Shape: (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
            Quantifies information transfer between variables.
        granger_causality : np.ndarray
            Granger causality measures. Shape: (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
            Indicates predictive causality based on past values.
        coeff_t : np.ndarray
            Estimated coefficients. Shape: (nobs, nvar, nvar * morder + 1).
        transfer_entropy_residual_cov : np.ndarray
            Residual covariance for transfer entropy. Shape: (T, 2) where T = nobs - morder.
    """
    nvar, nobs, ntrials = X.shape
    
    r = morder
    r1 = morder + 1
    extended_X = np.zeros((nvar, r1, nobs + r1 - 1, ntrials))
    
    for k in range(r1):
        extended_X[:, k, k:k + nobs, :] = X
    
    current_X = extended_X[:, 0, r1 - 1:nobs, :]
    lagged_X = extended_X[:, 1:r+1, r1 - 1:nobs - 1, :]
    # T = nobs - morder
    T = current_X.shape[1] - 1
    cov_Xp, cov_Yp, C_XYp, C_YXp = compute_covariances(lagged_X, T, morder)
    
    transfer_entropy_residual_cov = np.zeros((T, 2))
    transfer_entropy = np.zeros((T, 2))
    causal_strength = np.zeros((T, 2))
    granger_causality = np.zeros((T, 2))
        
    coeff_t = np.full((nobs, nvar, nvar * morder + 1), np.nan)
    
    for t in range(T):
        coefficients, residual_cov = estimate_coefficients(current_X[:, t, :].T,
                                                        lagged_X[:, :morder, t, :].reshape(nvar * morder, ntrials).T,
                                                        ntrials)
        coeff_t[t, :, :] = coefficients
        A_square = coefficients[:, :-1].reshape(nvar, nvar, morder)
        b = A_square[0, 1, :]
        c = A_square[1, 0, :]
        sigy = residual_cov[0, 0]
        sigx = residual_cov[1, 1]
        
        cov_Yp_reg = regularize_if_singular(cov_Yp[t])
        cov_Xp_reg = regularize_if_singular(cov_Xp[t])
        
        if time_mode == 'inhomo':
            transfer_entropy_residual_cov[t, 1] = (
                sigy + b.T @ cov_Xp_reg @ b 
                - b.T @ C_XYp[t] @ np.linalg.inv(cov_Yp_reg) @ C_XYp[t].T @ b
            )
            transfer_entropy_residual_cov[t, 0] = (
                sigx + c.T @ cov_Yp_reg @ c 
                - c.T @ C_YXp[t] @ np.linalg.inv(cov_Xp_reg) @ C_YXp[t].T @ c
            )

            transfer_entropy[t, 1] = 0.5 * np.log(transfer_entropy_residual_cov[t, 1] / sigy)
            transfer_entropy[t, 0] = 0.5 * np.log(transfer_entropy_residual_cov[t, 0] / sigx)
            
            if not diag_flag:
                causal_strength[t, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp_reg @ b) / sigy)
                causal_strength[t, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp_reg @ c) / sigx)
            else:
                causal_strength[t, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp_reg)) @ b) / sigy)
                causal_strength[t, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp_reg)) @ c) / sigx)
                
            
            lagged_X_p = lagged_X[1, :, t, :].T
            lagged_Y_p = lagged_X[0, :, t, :].T
            current_X_t = current_X[1, t, :]
            current_Y_t = current_X[0, t, :]
            _, sigx_r = estimate_coefficients(current_X_t, lagged_X_p, ntrials)
            _, sigy_r = estimate_coefficients(current_Y_t, lagged_Y_p, ntrials)
            granger_causality[t, 1] = np.log(sigy_r / sigy)
            granger_causality[t, 0] = np.log(sigx_r / sigx)
        
    if time_mode == 'homo':
        cov_Xp = np.mean(cov_Xp, axis=0)
        cov_Yp = np.mean(cov_Yp, axis=0)
        Sig_Xp = np.mean(Sig_Xp, axis=0)
        Sig_Yp = np.mean(Sig_Yp, axis=0)
        C_XYp = np.mean(C_XYp, axis=0)
        C_YXp = np.mean(C_YXp, axis=0)
        S_Xp = np.mean(S_Xp, axis=0)
        S_Yp = np.mean(S_Yp, axis=0)
        
        transfer_entropy = np.zeros((1, 2))
        causal_strength = np.zeros((1, 2))
        granger_causality = np.zeros((1, 2))
        
        cov_Yp_reg = regularize_if_singular(cov_Yp)
        cov_Xp_reg = regularize_if_singular(cov_Xp)
        
        transfer_entropy[0, 1] = 0.5 * np.log((sigy + b.T @ Sig_Xp @ b - b.T @ C_XYp @ np.linalg.inv(Sig_Yp) @ C_XYp.T @ b) / sigy)
        transfer_entropy[0, 0] = 0.5 * np.log((sigx + c.T @ Sig_Yp @ c - c.T @ C_YXp @ np.linalg.inv(Sig_Xp) @ C_YXp.T @ c) / sigx)
        
        if not diag_flag:
            causal_strength[0, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp_reg @ b) / sigy)
            causal_strength[0, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp_reg @ c) / sigx)
        else:
            causal_strength[0, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp_reg)) @ b) / sigy)
            causal_strength[0, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp_reg)) @ c) / sigx)
        
        granger_causality[0, 1] = np.log(sigy_r / sigy)
        granger_causality[0, 0] = np.log(sigx_r / sigx)
        
    if time_mode == 'inhomo':
        nan_block = np.full((morder, 2), np.nan)
        
        granger_causality = np.vstack([nan_block, granger_causality])
        transfer_entropy = np.vstack([nan_block, transfer_entropy])
        causal_strength = np.vstack([nan_block, causal_strength])

    return causal_strength, transfer_entropy, granger_causality, coeff_t, transfer_entropy_residual_cov

def time_varying_causality(Yt_event: np.ndarray, Yt_stats: Dict, CausalParams: Dict) -> Dict:
    _, nobs, ntrials = Yt_event.shape
    nvar = Yt_stats['OLS']['At'].shape[1]
    ref_time = CausalParams['ref_time']

    # Yt = Yt - mean_Yt
    # diag_flag = 0

    CausalOutput = {'TE': np.zeros((nobs, 2)),
                    'DCS': np.zeros((nobs, 2)),
                    'rDCS': np.zeros((nobs, 2))}

    for t in range(nobs):
        Xt = Yt_event[:2, t, :]  # Shape: (2, trials)
        Xp = Yt_event[2:, t, :]  # Shape: (nvar*mo, trials)

        if CausalParams['estim_mode'] == 'OLS':
            coeff = Yt_stats['OLS']['At'][t, :, :]
            SIG = Yt_stats['OLS']['Sigma_Et'][t, :, :]
            
        elif CausalParams['estim_mode'] == 'RLS':
            coeff = Yt_stats['RLS']['At'][t, :, :]
            SIG = Yt_stats['RLS']['Sigma_Et'][t, :, :]

        A_square = coeff.reshape(nvar, nvar, CausalParams['morder'])

        a = A_square[0, 0, :]  # Shape: (mo,)
        b = A_square[0, 1, :]  # Shape: (mo,)
        sigy = SIG[0, 0]  # Scalar
        c = A_square[1, 0, :]  # Shape: (mo,)
        d = A_square[1, 1, :]  # Shape: (mo,)
        sigx = SIG[1, 1]  # Scalar

        # from this line on X/Y means two variables (Y is the first)

        cov_Xp = Yt_stats['Sigma'][t, 3::2, 3::2]  # Shape: (mo, mo)
        cov_Yp = Yt_stats['Sigma'][t, 2::2, 2::2]  # Shape: (mo, mo)
        C_XYp = Yt_stats['Sigma'][t, 3::2, 2::2]  # Shape: (mo, mo)
        C_YXp = Yt_stats['Sigma'][t, 2::2, 3::2]  # Shape: (mo, mo)

        mean_Xp = Yt_stats['mean'][3::2, t]  # Shape: (mo,)
        mean_Yp = Yt_stats['mean'][2::2, t]  # Shape: (mo,)

        if sigy == 0:
            # Smallest positive float to avoid divide by zero
            sigy = np.finfo(float).eps
        if sigx == 0:
            sigx = np.finfo(float).eps
        
        cov_Xp = regularize_if_singular(cov_Xp)
        cov_Yp = regularize_if_singular(cov_Yp)
        
        CausalOutput['TE'][t, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp @ b - 
                                                b.T @ C_XYp @ np.linalg.inv(cov_Yp) @ C_XYp.T @ b) / sigy)
        CausalOutput['TE'][t, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp @ c - 
                                                c.T @ C_YXp @ np.linalg.inv(cov_Xp) @ C_YXp.T @ c) / sigx)

        mean_X_ref = np.mean(Yt_stats['mean'][3::2, ref_time], axis=1)  # Shape: (mo,)
        mean_Y_ref = np.mean(Yt_stats['mean'][2::2, ref_time], axis=1)  # Shape: (mo,)

        cov_Xp_ref = (cov_Xp + mean_Xp[:, np.newaxis] @ mean_Xp[np.newaxis, :] - 
                      mean_Xp[:, np.newaxis] @ mean_X_ref[np.newaxis, :] - 
                      mean_X_ref[:, np.newaxis] @ mean_Xp[np.newaxis, :] + 
                      mean_X_ref[:, np.newaxis] @ mean_X_ref[np.newaxis, :])
        cov_Yp_ref = (cov_Yp + mean_Yp[:, np.newaxis] @ mean_Yp[np.newaxis, :] - 
                      mean_Yp[:, np.newaxis] @ mean_Y_ref[np.newaxis, :] - 
                      mean_Y_ref[:, np.newaxis] @ mean_Yp[np.newaxis, :] + 
                      mean_Y_ref[:, np.newaxis] @ mean_Y_ref[np.newaxis, :])

        # cov_Xp_lag = (
        #                 (Xp - np.mean(mean_Yt[2:, :ref_time], axis=1, keepdims=True))
        #                 @ (Xp - np.mean(mean_Yt[2:, :ref_time], axis=1, keepdims=True)).T
        #             ) / ntrials

        ref_cov_Xp = np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0)
        ref_cov_Yp = np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0)
            
        if not CausalParams['diag_flag']:
            CausalOutput['DCS'][t, 1] = 0.5 * np.log((sigy + b.T @ cov_Xp @ b) / sigy)
            CausalOutput['DCS'][t, 0] = 0.5 * np.log((sigx + c.T @ cov_Yp @ c) / sigx)
                
            if CausalParams['old_version']:
                cov_Xp_lag = np.dot(Xp - np.mean(Yt_stats['mean'][2:, ref_time], axis=1)[:, np.newaxis],
                            (Xp - np.mean(Yt_stats['mean'][2:, ref_time], axis=1)[:, np.newaxis]).T) / ntrials

                CausalOutput['rDCS'][t, 1] = (0.5 * np.log((sigy + b.T @ ref_cov_Xp @ b) / sigy) - 0.5 +
                                             0.5 * (sigy + b.T @ cov_Xp_lag[1::2, 1::2] @ b) / 
                                             (sigy + b.T @ ref_cov_Xp @ b))
                CausalOutput['rDCS'][t, 0] = (0.5 * np.log((sigx + c.T @ ref_cov_Yp @ c) / sigx) - 0.5 +
                                             0.5 * (sigx + c.T @ cov_Xp_lag[0::2, 0::2] @ c) / 
                                             (sigx + c.T @ ref_cov_Yp @ c))
            else:
                CausalOutput['rDCS'][t, 1] = (0.5 * np.log((sigy + b.T @ ref_cov_Xp @ b) / sigy) - 0.5 +
                                             0.5 * (sigy + b.T @ cov_Xp_ref @ b) / (sigy + b.T @ ref_cov_Xp @ b))
                CausalOutput['rDCS'][t, 0] = (0.5 * np.log((sigx + c.T @ ref_cov_Yp @ c) / sigx) - 0.5 +
                                             0.5 * (sigx + c.T @ cov_Yp_ref @ c) / (sigx + c.T @ ref_cov_Yp @ c))
        else:
            CausalOutput['DCS'][t, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(cov_Xp)) @ b) / sigy)
            CausalOutput['DCS'][t, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(cov_Yp)) @ c) / sigx)
            
            if CausalParams['old_version']:
                cov_Xp_lag = np.dot(Xp - np.mean(Yt_stats['mean'][2:, ref_time], axis=1)[:, np.newaxis],
                            (Xp - np.mean(Yt_stats['mean'][2:, ref_time], axis=1)[:, np.newaxis]).T) / ntrials
                    
                CausalOutput['rDCS'][t, 1] = (0.5 * np.log((sigy + b.T @ np.diag(np.diag(ref_cov_Xp)) @ b) / sigy) - 0.5 +
                                             0.5 * (sigy + b.T @ np.diag(np.diag(cov_Xp_lag[1::2, 1::2])) @ b) / 
                                             (sigy + b.T @ np.diag(np.diag(ref_cov_Xp)) @ b))
                CausalOutput['rDCS'][t, 0] = (0.5 * np.log((sigx + c.T @ np.diag(np.diag(ref_cov_Yp)) @ c) / sigx) - 0.5 +
                                             0.5 * (sigx + c.T @ np.diag(np.diag(cov_Xp_lag[0::2, 0::2])) @ c) / 
                                             (sigx + c.T @ np.diag(np.diag(ref_cov_Yp)) @ c))
                
            else:
                CausalOutput['rDCS'][t, 1] = (0.5 * np.log((sigy + b.T @ np.diag(np.diag(ref_cov_Xp)) @ b) / sigy) - 0.5 +
                                             0.5 * (sigy + b.T @ np.diag(np.diag(cov_Xp_ref)) @ b) / 
                                             (sigy + b.T @ np.diag(np.diag(ref_cov_Xp)) @ b))
                CausalOutput['rDCS'][t, 0] = (0.5 * np.log((sigx + c.T @ np.diag(np.diag(ref_cov_Yp)) @ c) / sigx) - 0.5 +
                                             0.5 * (sigx + c.T @ np.diag(np.diag(cov_Yp_ref)) @ c) / 
                                             (sigx + c.T @ np.diag(np.diag(ref_cov_Yp)) @ c))

    return CausalOutput
