import numpy as np


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
        Xt = Yt_event[0:1, t, :]
        Xp = Yt_event[2:, t, :]

        if CausalParams['estim_mode'] == 'OLS':
            coeff = Yt_stats['OLS']['At'][t, :, :]
            SIG = Yt_stats['OLS']['Sigma_Et'][t, :, :]
        elif CausalParams['estim_mode'] == 'RLS':
            coeff = Yt_stats['RLS']['At'][t, :, :]
            SIG = Yt_stats['RLS']['Sigma_Et'][t, :, :]

        A_square = coeff.reshape(nvar, nvar, CausalParams['morder'])

        a, b = A_square[0, 0, :], A_square[0, 1, :]
        sigy = SIG[0, 0]

        c, d = A_square[1, 0, :], A_square[1, 1, :]
        sigx = SIG[1, 1]

        # from this line on X/Y means two variables (Y is the first)

        cov_Xp = Yt_stats['Sigma'][t, 3::2, 3::2]
        cov_Yp = Yt_stats['Sigma'][t, 2::2, 2::2]
        C_XYp = Yt_stats['Sigma'][t, 3::2, 2::2]
        C_YXp = Yt_stats['Sigma'][t, 2::2, 3::2]

        mean_Xp = Yt_stats['mean'][3::2, t]
        mean_Yp = Yt_stats['mean'][2::2, t]

        if sigy == 0:
            # Smallest positive float to avoid divide by zero
            sigy = np.finfo(float).eps
        if sigx == 0:
            sigx = np.finfo(float).eps
            
        CausalOutput['TE'][t, 1] = 0.5 * np.log(
            (sigy + b.T @ cov_Xp @ b - b.T @ C_XYp @ np.linalg.inv(cov_Yp) @ C_XYp.T @ b) / sigy)
        CausalOutput['TE'][t, 0] = 0.5 * np.log(
            (sigx + c.T @ cov_Yp @ c - c.T @ C_YXp @ np.linalg.inv(cov_Xp) @ C_YXp.T @ c) / sigx)

        mean_X_ref = np.mean(Yt_stats['mean'][3::2, ref_time], axis=1)
        mean_Y_ref = np.mean(Yt_stats['mean'][2::2, ref_time], axis=1)

        cov_Xp_ref = cov_Xp + np.outer(mean_Xp, mean_Xp) - np.outer(
            mean_Xp, mean_X_ref) - np.outer(mean_X_ref, mean_Xp) + np.outer(mean_X_ref, mean_X_ref)
        cov_Yp_ref = cov_Yp + np.outer(mean_Yp, mean_Yp) - np.outer(
            mean_Yp, mean_Y_ref) - np.outer(mean_Y_ref, mean_Yp) + np.outer(mean_Y_ref, mean_Y_ref)

        # cov_Xp_lag = (
        #                 (Xp - np.mean(mean_Yt[2:, :ref_time], axis=1, keepdims=True))
        #                 @ (Xp - np.mean(mean_Yt[2:, :ref_time], axis=1, keepdims=True)).T
        #             ) / ntrials

        cov_Xp_lag = np.dot(Xp - np.mean(Yt_stats['mean'][2:, ref_time], axis=1)[:, np.newaxis],
                            (Xp - np.mean(Yt_stats['mean'][2:, ref_time], axis=1)[:, np.newaxis]).T) / ntrials

        if not CausalParams['diag_flag']:
            CausalOutput['DCS'][t, 1] = 0.5 * \
                np.log((sigy + b.T @ cov_Xp @ b) / sigy)
            CausalOutput['DCS'][t, 0] = 0.5 * \
                np.log((sigx + c.T @ cov_Yp @ c) / sigx)

            if CausalParams['old_version']:
                CausalOutput['rDCS'][t, 1] = 0.5 * np.log((sigy + b.T @ np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0) @ b) / sigy) - 0.5 + \
                        0.5 * (sigy + b.T @ cov_Xp_lag[1::2, 1::2] @ b) / (
                        sigy + b.T @ np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0) @ b)

                CausalOutput['rDCS'][t, 0] = 0.5 * np.log((sigx + c.T @ np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0) @ c) / sigx) - 0.5 + \
                    0.5 * (sigx + c.T @ cov_Xp_lag[0::2, 0::2] @ c) / (
                        sigx + c.T @ np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0) @ c)
            else:
                CausalOutput['rDCS'][t, 1] = 0.5 * np.log((sigy + b.T @ np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0) @ b) / sigy) - 0.5 + \
                    0.5 * (sigy + b.T @ cov_Xp_ref @ b) / (sigy + b.T @
                                                           np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0) @ b)

                CausalOutput['rDCS'][t, 0] = 0.5 * np.log((sigx + c.T @ np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0) @ c) / sigx) - 0.5 + \
                    0.5 * (sigx + c.T @ cov_Yp_ref @ c) / (sigx + c.T @
                                                           np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0) @ c)
        else:
            CausalOutput['DCS'][t, 1] = 0.5 * \
                np.log((sigy + b.T @ np.diag(np.diag(cov_Xp)) @ b) / sigy)
            CausalOutput['DCS'][t, 0] = 0.5 * \
                np.log((sigx + c.T @ np.diag(np.diag(cov_Yp)) @ c) / sigx)

            if CausalParams['old_version']:
                CausalOutput['rDCS'][t, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0))) @ b) / sigy) - 0.5 + \
                    0.5 * (sigy + b.T @ np.diag(np.diag(cov_Xp_lag[1::2, 1::2])) @ b) / (
                        sigy + b.T @ np.diag(np.diag(np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0))) @ b)

                CausalOutput['rDCS'][t, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0))) @ c) / sigx) - 0.5 + \
                    0.5 * (sigx + c.T @ np.diag(np.diag(cov_Xp_lag[0::2, 0::2])) @ c) / (
                        sigx + c.T @ np.diag(np.diag(np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0))) @ c)
            else:
                CausalOutput['rDCS'][t, 1] = 0.5 * np.log((sigy + b.T @ np.diag(np.diag(np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0))) @ b) / sigy) - 0.5 + \
                    0.5 * (sigy + b.T @ np.diag(np.diag(cov_Xp_ref)) @ b) / (sigy + b.T @ np.diag(
                        np.diag(np.mean(Yt_stats['Sigma'][ref_time, 3::2, 3::2], axis=0))) @ b)

                CausalOutput['rDCS'][t, 0] = 0.5 * np.log((sigx + c.T @ np.diag(np.diag(np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0))) @ c) / sigx) - 0.5 + \
                    0.5 * (sigx + c.T @ np.diag(np.diag(cov_Yp_ref)) @ c) / (sigx + c.T @ np.diag(
                        np.diag(np.mean(Yt_stats['Sigma'][ref_time, 2::2, 2::2], axis=0))) @ c)

    return CausalOutput
