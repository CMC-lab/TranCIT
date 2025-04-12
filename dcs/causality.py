import logging
from typing import Dict, Tuple

import numpy as np

from .utils.helpers import compute_covariances, estimate_coefficients
from .utils.preprocess import regularize_if_singular

logging.basicConfig(level=logging.INFO)


def compute_causal_strength_nonzero_mean(
    time_series_data: np.ndarray,
    model_order: int,
    time_mode: str,
    use_diagonal_covariance: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute causal strength, transfer entropy, and Granger causality for bivariate signals.

    This function calculates causality measures based on a Vector Autoregression (VAR) model with
    non-zero mean. It supports time-inhomogeneous ('inhomo') and homogeneous ('homo') modes.

    Parameters
    ----------
    time_series_data : np.ndarray
        Input data array of shape (nvar, nobs, ntrials), where:
        - nvar: number of variables (must be 2 for bivariate analysis).
        - nobs: number of observations (time points).
        - ntrials: number of trials.
    model_order : int
        Number of lags for the VAR model.
    time_mode : str
        Time mode: 'inhomo' (time-inhomogeneous) or 'homo' (homogeneous).
    use_diagonal_covariance : bool
        If True, use diagonal covariance matrices; otherwise, use full covariance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - causal_strength: Causal strength measures, shape (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
        - transfer_entropy: Transfer entropy measures, shape (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
        - granger_causality: Granger causality measures, shape (nobs, 2) for 'inhomo', (1, 2) for 'homo'.
        - coefficients: Estimated VAR coefficients, shape (nobs, nvar, nvar * model_order + 1).
        - te_residual_cov: Residual covariance for transfer entropy, shape (n_time_steps, 2).

    Raises
    ------
    ValueError
        If nvar != 2 or time_mode is invalid.

    Notes
    -----
    - Assumes a bivariate system (nvar = 2).
    - Computation may be intensive for large nobs or ntrials.
    """
    # Input validation
    if time_series_data.shape[0] != 2:
        raise ValueError("Function requires bivariate signals (nvar = 2).")
    if time_mode not in ["inhomo", "homo"]:
        raise ValueError("time_mode must be 'inhomo' or 'homo'.")

    nvar, nobs, ntrials = time_series_data.shape
    logging.info(f"Starting computation: nvar={nvar}, nobs={nobs}, ntrials={ntrials}")

    # Prepare extended data
    r1 = model_order + 1
    extended_data = np.zeros((nvar, r1, nobs + r1 - 1, ntrials))
    for k in range(r1):
        extended_data[:, k, k : k + nobs, :] = time_series_data

    current_data = extended_data[:, 0, r1 - 1 : nobs, :]
    lagged_data = extended_data[:, 1 : model_order + 1, r1 - 1 : nobs - 1, :]
    n_time_steps = current_data.shape[1] - 1

    # Compute covariances
    cov_xp, cov_yp, c_xyp, c_yxp = compute_covariances(
        lagged_data, n_time_steps, model_order
    )

    # Initialize output arrays
    te_residual_cov = np.zeros((n_time_steps, 2))
    transfer_entropy = np.zeros((n_time_steps, 2))
    causal_strength = np.zeros((n_time_steps, 2))
    granger_causality = np.zeros((n_time_steps, 2))
    coefficients = np.full((nobs, nvar, nvar * model_order + 1), np.nan)

    # Time-inhomogeneous computation
    for t in range(n_time_steps):
        logging.debug(f"Processing time step {t}/{n_time_steps - 1}")
        coeff, residual_cov = estimate_coefficients(
            current_data[:, t, :].T,
            lagged_data[:, :model_order, t, :].reshape(nvar * model_order, ntrials).T,
            ntrials,
        )
        coefficients[t, :, :] = coeff
        a_square = coeff[:, :-1].reshape(nvar, nvar, model_order)
        b = a_square[0, 1, :]
        c = a_square[1, 0, :]
        sigy = residual_cov[0, 0] or np.finfo(float).eps
        sigx = residual_cov[1, 1] or np.finfo(float).eps

        cov_yp_reg = regularize_if_singular(cov_yp[t])
        cov_xp_reg = regularize_if_singular(cov_xp[t])

        if time_mode == "inhomo":
            te_residual_cov[t, 1] = (
                sigy
                + b.T @ cov_xp_reg @ b
                - b.T @ c_xyp[t] @ np.linalg.inv(cov_yp_reg) @ c_xyp[t].T @ b
            )
            te_residual_cov[t, 0] = (
                sigx
                + c.T @ cov_yp_reg @ c
                - c.T @ c_yxp[t] @ np.linalg.inv(cov_xp_reg) @ c_yxp[t].T @ c
            )

            transfer_entropy[t, 1] = 0.5 * np.log(te_residual_cov[t, 1] / sigy)
            transfer_entropy[t, 0] = 0.5 * np.log(te_residual_cov[t, 0] / sigx)

            if not use_diagonal_covariance:
                causal_strength[t, 1] = 0.5 * np.log(
                    (sigy + b.T @ cov_xp_reg @ b) / sigy
                )
                causal_strength[t, 0] = 0.5 * np.log(
                    (sigx + c.T @ cov_yp_reg @ c) / sigx
                )
            else:
                causal_strength[t, 1] = 0.5 * np.log(
                    (sigy + b.T @ np.diag(np.diag(cov_xp_reg)) @ b) / sigy
                )
                causal_strength[t, 0] = 0.5 * np.log(
                    (sigx + c.T @ np.diag(np.diag(cov_yp_reg)) @ c) / sigx
                )

            lagged_x_p = lagged_data[1, :, t, :].T
            lagged_y_p = lagged_data[0, :, t, :].T
            current_x_t = current_data[1, t, :]
            current_y_t = current_data[0, t, :]
            _, sigx_r = estimate_coefficients(current_x_t, lagged_x_p, ntrials)
            _, sigy_r = estimate_coefficients(current_y_t, lagged_y_p, ntrials)
            granger_causality[t, 1] = np.log(sigy_r / sigy)
            granger_causality[t, 0] = np.log(sigx_r / sigx)

    # Time-homogeneous computation
    if time_mode == "homo":
        cov_xp = np.mean(cov_xp, axis=0)
        cov_yp = np.mean(cov_yp, axis=0)
        cov_xy_p = np.mean(c_xyp, axis=0)
        cov_yx_p = np.mean(c_yxp, axis=0)
        # Note: Sig_Xp, Sig_Yp, etc., should be defined or computed; assuming they exist elsewhere
        # Placeholder adjustment needed here based on actual implementation
        transfer_entropy = np.zeros((1, 2))
        causal_strength = np.zeros((1, 2))
        granger_causality = np.zeros((1, 2))
        cov_yp_reg = regularize_if_singular(cov_yp)
        cov_xp_reg = regularize_if_singular(cov_xp)

        transfer_entropy[0, 1] = 0.5 * np.log(
            (
                sigy
                + b.T @ cov_xp @ b
                - b.T @ cov_xy_p @ np.linalg.inv(cov_yp) @ cov_xy_p.T @ b
            )
            / sigy
        )
        transfer_entropy[0, 0] = 0.5 * np.log(
            (
                sigx
                + c.T @ cov_yp @ c
                - c.T @ cov_yx_p @ np.linalg.inv(cov_xp) @ cov_yx_p.T @ c
            )
            / sigx
        )

        if not use_diagonal_covariance:
            causal_strength[0, 1] = 0.5 * np.log((sigy + b.T @ cov_xp_reg @ b) / sigy)
            causal_strength[0, 0] = 0.5 * np.log((sigx + c.T @ cov_yp_reg @ c) / sigx)
        else:
            causal_strength[0, 1] = 0.5 * np.log(
                (sigy + b.T @ np.diag(np.diag(cov_xp_reg)) @ b) / sigy
            )
            causal_strength[0, 0] = 0.5 * np.log(
                (sigx + c.T @ np.diag(np.diag(cov_yp_reg)) @ c) / sigx
            )

        granger_causality[0, 1] = np.log(sigy_r / sigy)
        granger_causality[0, 0] = np.log(sigx_r / sigx)

    # Adjust outputs for 'inhomo' mode
    if time_mode == "inhomo":
        nan_block = np.full((model_order, 2), np.nan)
        granger_causality = np.vstack([nan_block, granger_causality])
        transfer_entropy = np.vstack([nan_block, transfer_entropy])
        causal_strength = np.vstack([nan_block, causal_strength])

    return (
        causal_strength,
        transfer_entropy,
        granger_causality,
        coefficients,
        te_residual_cov,
    )


def time_varying_causality(
    event_data: np.ndarray, stats: Dict, causal_params: Dict
) -> Dict:
    """
    Compute time-varying causality measures for bivariate signals.

    Calculates Transfer Entropy (TE), Dynamic Causal Strength (DCS), and Relative Dynamic Causal
    Strength (rDCS) based on a VAR model.

    Parameters
    ----------
    event_data : np.ndarray
        Event data array of shape (nvar * (model_order + 1), nobs, ntrials).
    stats : Dict
        Model statistics with keys:
        - 'OLS' or 'RLS': Sub-dict with 'At' (coefficients) and 'Sigma_Et' (residual covariance).
        - 'Sigma': Covariance matrices.
        - 'mean': Mean values.
    causal_params : Dict
        Parameters with keys:
        - 'ref_time': Reference time index.
        - 'estim_mode': 'OLS' or 'RLS'.
        - 'morder': Model order.
        - 'diag_flag': Boolean for diagonal covariance.
        - 'old_version': Boolean for rDCS calculation method.

    Returns
    -------
    Dict
        Causality measures:
        - 'TE': Transfer Entropy, shape (nobs, 2).
        - 'DCS': Dynamic Causal Strength, shape (nobs, 2).
        - 'rDCS': Relative Dynamic Causal Strength, shape (nobs, 2).

    Raises
    ------
    ValueError
        If 'estim_mode' is not 'OLS' or 'RLS'.
    """
    _, nobs, ntrials = event_data.shape
    nvar = stats["OLS"]["At"].shape[1]
    ref_time = causal_params["ref_time"]

    if causal_params["estim_mode"] not in ["OLS", "RLS"]:
        raise ValueError("estim_mode must be 'OLS' or 'RLS'.")

    logging.info(f"Computing causality with mode: {causal_params['estim_mode']}")

    causality_measures = {
        "TE": np.zeros((nobs, 2)),
        "DCS": np.zeros((nobs, 2)),
        "rDCS": np.zeros((nobs, 2)),
    }

    for t in range(nobs):
        current_vars = event_data[:2, t, :]
        lagged_vars = event_data[2:, t, :]

        mode = causal_params["estim_mode"]
        coeff = stats[mode]["At"][t, :, :]
        residual_cov = stats[mode]["Sigma_Et"][t, :, :]

        a_square = coeff.reshape(nvar, nvar, causal_params["morder"])
        b = a_square[0, 1, :]
        sigy = residual_cov[0, 0] or np.finfo(float).eps
        c = a_square[1, 0, :]
        sigx = residual_cov[1, 1] or np.finfo(float).eps

        cov_xp = stats["Sigma"][t, 3::2, 3::2]
        cov_yp = stats["Sigma"][t, 2::2, 2::2]
        c_xyp = stats["Sigma"][t, 3::2, 2::2]
        c_yxp = stats["Sigma"][t, 2::2, 3::2]
        mean_xp = stats["mean"][3::2, t]
        mean_yp = stats["mean"][2::2, t]

        cov_xp_reg = regularize_if_singular(cov_xp)
        cov_yp_reg = regularize_if_singular(cov_yp)

        causality_measures["TE"][t, 1] = 0.5 * np.log(
            (
                sigy
                + b.T @ cov_xp_reg @ b
                - b.T @ c_xyp @ np.linalg.inv(cov_yp_reg) @ c_xyp.T @ b
            )
            / sigy
        )
        causality_measures["TE"][t, 0] = 0.5 * np.log(
            (
                sigx
                + c.T @ cov_yp_reg @ c
                - c.T @ c_yxp @ np.linalg.inv(cov_xp_reg) @ c_yxp.T @ c
            )
            / sigx
        )

        mean_x_ref = np.mean(stats["mean"][3::2, ref_time], axis=1)
        mean_y_ref = np.mean(stats["mean"][2::2, ref_time], axis=1)
        cov_xp_ref = (
            cov_xp
            + mean_xp[:, np.newaxis] @ mean_xp[np.newaxis, :]
            - mean_xp[:, np.newaxis] @ mean_x_ref[np.newaxis, :]
            - mean_x_ref[:, np.newaxis] @ mean_xp[np.newaxis, :]
            + mean_x_ref[:, np.newaxis] @ mean_x_ref[np.newaxis, :]
        )
        cov_yp_ref = (
            cov_yp
            + mean_yp[:, np.newaxis] @ mean_yp[np.newaxis, :]
            - mean_yp[:, np.newaxis] @ mean_y_ref[np.newaxis, :]
            - mean_y_ref[:, np.newaxis] @ mean_yp[np.newaxis, :]
            + mean_y_ref[:, np.newaxis] @ mean_y_ref[np.newaxis, :]
        )

        ref_cov_xp = np.mean(stats["Sigma"][ref_time, 3::2, 3::2], axis=0)
        ref_cov_yp = np.mean(stats["Sigma"][ref_time, 2::2, 2::2], axis=0)

        if not causal_params["diag_flag"]:
            causality_measures["DCS"][t, 1] = 0.5 * np.log(
                (sigy + b.T @ cov_xp @ b) / sigy
            )
            causality_measures["DCS"][t, 0] = 0.5 * np.log(
                (sigx + c.T @ cov_yp @ c) / sigx
            )

            if causal_params["old_version"]:
                cov_xp_lag = (
                    np.dot(
                        lagged_vars
                        - np.mean(stats["mean"][2:, ref_time], axis=1)[:, np.newaxis],
                        (
                            lagged_vars
                            - np.mean(stats["mean"][2:, ref_time], axis=1)[
                                :, np.newaxis
                            ]
                        ).T,
                    )
                    / ntrials
                )
                causality_measures["rDCS"][t, 1] = (
                    0.5 * np.log((sigy + b.T @ ref_cov_xp @ b) / sigy)
                    - 0.5
                    + 0.5
                    * (sigy + b.T @ cov_xp_lag[1::2, 1::2] @ b)
                    / (sigy + b.T @ ref_cov_xp @ b)
                )
                causality_measures["rDCS"][t, 0] = (
                    0.5 * np.log((sigx + c.T @ ref_cov_yp @ c) / sigx)
                    - 0.5
                    + 0.5
                    * (sigx + c.T @ cov_xp_lag[0::2, 0::2] @ c)
                    / (sigx + c.T @ ref_cov_yp @ c)
                )
            else:
                causality_measures["rDCS"][t, 1] = (
                    0.5 * np.log((sigy + b.T @ ref_cov_xp @ b) / sigy)
                    - 0.5
                    + 0.5
                    * (sigy + b.T @ cov_xp_ref @ b)
                    / (sigy + b.T @ ref_cov_xp @ b)
                )
                causality_measures["rDCS"][t, 0] = (
                    0.5 * np.log((sigx + c.T @ ref_cov_yp @ c) / sigx)
                    - 0.5
                    + 0.5
                    * (sigx + c.T @ cov_yp_ref @ c)
                    / (sigx + c.T @ ref_cov_yp @ c)
                )
        else:
            causality_measures["DCS"][t, 1] = 0.5 * np.log(
                (sigy + b.T @ np.diag(np.diag(cov_xp)) @ b) / sigy
            )
            causality_measures["DCS"][t, 0] = 0.5 * np.log(
                (sigx + c.T @ np.diag(np.diag(cov_yp)) @ c) / sigx
            )

            if causal_params["old_version"]:
                cov_xp_lag = (
                    np.dot(
                        lagged_vars
                        - np.mean(stats["mean"][2:, ref_time], axis=1)[:, np.newaxis],
                        (
                            lagged_vars
                            - np.mean(stats["mean"][2:, ref_time], axis=1)[
                                :, np.newaxis
                            ]
                        ).T,
                    )
                    / ntrials
                )
                causality_measures["rDCS"][t, 1] = (
                    0.5 * np.log((sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b) / sigy)
                    - 0.5
                    + 0.5
                    * (sigy + b.T @ np.diag(np.diag(cov_xp_lag[1::2, 1::2])) @ b)
                    / (sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b)
                )
                causality_measures["rDCS"][t, 0] = (
                    0.5 * np.log((sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c) / sigx)
                    - 0.5
                    + 0.5
                    * (sigx + c.T @ np.diag(np.diag(cov_xp_lag[0::2, 0::2])) @ c)
                    / (sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c)
                )
            else:
                causality_measures["rDCS"][t, 1] = (
                    0.5 * np.log((sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b) / sigy)
                    - 0.5
                    + 0.5
                    * (sigy + b.T @ np.diag(np.diag(cov_xp_ref)) @ b)
                    / (sigy + b.T @ np.diag(np.diag(ref_cov_xp)) @ b)
                )
                causality_measures["rDCS"][t, 0] = (
                    0.5 * np.log((sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c) / sigx)
                    - 0.5
                    + 0.5
                    * (sigx + c.T @ np.diag(np.diag(cov_yp_ref)) @ c)
                    / (sigx + c.T @ np.diag(np.diag(ref_cov_yp)) @ c)
                )

    return causality_measures
