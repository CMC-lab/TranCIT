import numpy as np
from dcs.utils.preprocessing.singular import regularize_if_singular

def time_varying_causality(Yt_event, Yt_stats, CausalParams):
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
