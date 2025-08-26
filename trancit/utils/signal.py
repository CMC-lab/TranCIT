import logging
from typing import Tuple

import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


def find_best_shrinked_locations(
    signal: np.ndarray,
    shrinked_locations: np.ndarray,
    all_locations: np.ndarray,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the best subset of shrinked locations by minimizing histogram distance.

    Computes the histogram of the full set of locations (`all_locations`) and compares it to
    histograms of subsets of `shrinked_locations` using Euclidean distance. Returns the subset
    that minimizes this distance.

    Args:
        signal (np.ndarray): 1D array of signal or data values.
        shrinked_locations (np.ndarray): 1D array of subsampled location indices.
        all_locations (np.ndarray): 1D array of all original location indices.
        num_bins (int, optional): Number of histogram bins. Defaults to 100.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - best_locations: Subset of `shrinked_locations` with minimal histogram distance.
            - distances: Array of distances for each subset size, with NaN for sizes < 100.

    Raises:
        ValueError: If input arrays are empty or incompatible with `num_bins`.
    """
    if not signal.size or not shrinked_locations.size or not all_locations.size:
        logger.error("Empty input arrays provided to find_best_shrinked_locs.")
        raise ValueError("Signal and location arrays must not be empty.")

    hist_full, _ = np.histogram(signal[all_locations], bins=num_bins, density=True)
    logger.debug(f"Computed full histogram with {num_bins} bins.")

    distances = np.full(len(shrinked_locations), np.nan)

    for size in range(100, len(shrinked_locations)):
        hist_temp, _ = np.histogram(
            signal[shrinked_locations[:size]], bins=num_bins, density=True
        )
        # distances[size] = cdist(
        #     hist_full.reshape(1, -1), hist_temp.reshape(1, -1), metric="euclidean"
        # )[0, 0]
        distances[size] = euclidean(hist_full, hist_temp)
        logger.debug(f"Computed distance {distances[size]:.4f} for subset size {size}.")

    best_size = np.nanargmin(distances)
    best_locations = shrinked_locations[:best_size]
    logger.info(
        f"Best subset size: {best_size} with distance {distances[best_size]:.4f}."
    )

    return best_locations, distances


def shrink_locations_resample_uniform(
    locations: np.ndarray, min_distance: int
) -> np.ndarray:
    """
    Uniformly resample locations with a minimum distance between points.

    Iteratively selects locations randomly, ensuring each new location is at least `min_distance`
    away from previously selected ones, until no more can be added.

    Args:
        locations (np.ndarray): 1D array of location indices to resample from.
        min_distance (int): Minimum distance between selected locations.

    Returns:
        np.ndarray: 1D array of selected locations.

    Raises:
        ValueError: If `min_distance` is negative or `locations` is empty.
    """
    if len(locations) == 0:
        logger.warning("Empty locations array provided; returning empty array.")
        return np.array([])
    if min_distance < 0:
        logger.error("Negative min_distance provided.")
        raise ValueError("min_distance must be non-negative.")

    selected_locations: list[int] = []
    available_locations = locations.copy()
    max_number_generation = len(available_locations)
    logger.debug(
        f"Starting resampling with {len(locations)} locations and min_distance {min_distance}."
    )

    while len(selected_locations) < max_number_generation:
        rand_idx = np.random.randint(0, len(available_locations))
        selected_loc = available_locations[rand_idx]
        selected_locations.append(selected_loc)
        mask = np.abs(selected_loc - available_locations) >= min_distance
        available_locations = available_locations[mask]
        logger.debug(
            f"Selected location {selected_loc}; {len(available_locations)} remain."
        )

    result = np.array(selected_locations)
    logger.info(f"Resampling complete; selected {len(result)} locations.")
    return result


def find_peak_locations(
    signal: np.ndarray, candidate_locations: np.ndarray, window_size: int
) -> np.ndarray:
    """
    Refine peak locations by grouping candidates and local maximization.

    Groups candidate locations within `window_size`, finds the maximum in each group, and refines
    the peak within a local window around each maximum.

    Args:
        signal (np.ndarray): 1D array of signal data.
        candidate_locations (np.ndarray): 1D array of candidate peak indices.
        window_size (int): Size of the window for grouping and refinement.

    Returns:
        np.ndarray: 1D array of refined peak locations.

    Raises:
        ValueError: If `window_size` is negative or inputs are empty/invalid.
    """
    if not signal.size or not candidate_locations.size:
        logger.error("Empty signal or candidate_locations provided.")
        raise ValueError("Signal and candidate_locations must not be empty.")
    if window_size < 0:
        logger.error("Negative window_size provided.")
        raise ValueError("window_size must be non-negative.")

    valid_candidates = np.sort(
        candidate_locations[
            (candidate_locations >= window_size)
            & (candidate_locations <= len(signal) - window_size)
        ].astype(np.intp)
    )
    logger.debug(f"Filtered to {len(valid_candidates)} valid candidates.")

    preliminary_peaks = []
    idx_start = 0
    idx_end = 0
    while idx_end < len(valid_candidates) - 1:
        while (
            idx_end + 1 < len(valid_candidates)
            and valid_candidates[idx_end + 1] - valid_candidates[idx_start]
            < window_size
        ):
            idx_end += 1
        if idx_end == len(valid_candidates) - 1:
            break

        group = valid_candidates[idx_start : idx_end + 1]
        group_signal = signal[group]
        max_idx = np.argmax(group_signal)
        preliminary_peaks.append(group[max_idx])
        idx_start = idx_end + 1
        idx_end = idx_start
    logger.debug(f"Found {len(preliminary_peaks)} preliminary peaks.")

    # Refine peaks within local windows
    half_window = int(np.ceil(window_size / 2))
    refined_peaks = []
    for peak in preliminary_peaks:
        start = peak - half_window + 1
        end = peak + half_window
        if start < 0 or end > len(signal):
            logger.warning(f"Skipping peak at {peak} due to boundary violation.")
            continue
        local_signal = signal[start:end]
        local_max_idx = np.argmax(local_signal)
        refined_peaks.append(start + local_max_idx)
    logger.info(f"Refined to {len(refined_peaks)} peaks.")

    return np.unique(refined_peaks)
