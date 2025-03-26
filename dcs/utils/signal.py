import numpy as np
from scipy.spatial.distance import cdist, euclidean
from typing import Tuple


def find_best_shrinked_locs(D: np.ndarray, shrinked_locs: np.ndarray, all_locs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    Nfull, _ = np.histogram(D[all_locs], bins=100, density=True)

    distance = np.full(len(shrinked_locs), np.nan)
    
    for n in range(100, len(shrinked_locs)):
        Ntemp, _ = np.histogram(D[shrinked_locs[:n]], bins=100, density=True)
        # distance[n] = euclidean(Nfull, Ntemp)
        distance[n] = cdist(Nfull.reshape(1, -1), Ntemp.reshape(1, -1))[0, 0]

    best_n = np.nanargmin(distance)
    best_locs = shrinked_locs[:best_n]

    return best_locs, distance


def shrink_locs_resample_uniform(loc: np.ndarray, L: int) -> np.ndarray:
    Ngen = 0
    maxNgen = len(loc)
    shrinked_locs = np.full(maxNgen, np.nan)
    loc_range = np.copy(loc)

    while Ngen < maxNgen:
        if loc_range.size == 0:
            break
        
        rand_idx = np.random.randint(0, len(loc_range))
        selected_loc = loc_range[rand_idx]
        shrinked_locs[Ngen] = selected_loc
        
        mask = np.abs(selected_loc - loc_range) >= L
        loc_range = loc_range[mask]

        Ngen += 1

    shrinked_locs = shrinked_locs[~np.isnan(shrinked_locs)]

    return shrinked_locs


def find_peak_loc(signal: np.ndarray, loc: np.ndarray, L: int) -> np.ndarray:
    loc = loc[(loc >= L) & (loc <= len(signal) - L)] # loc[loc <= len(signal) - L - 1]

    peak_loc1 = []
    idx_start = 0
    idx_end = 0
    
    while idx_end < len(loc) - 1:
        while loc[idx_end + 1] - loc[idx_start] < L:
            idx_end += 1
            if idx_end == len(loc) - 1:
                break

        temp = np.arange(idx_start, idx_end)
        temp_signal = signal[loc[temp]]
        temp_idx = np.argmax(temp_signal)
        peak_loc1.append(loc[temp_idx + idx_start - 1])

        idx_start = idx_end + 1
        idx_end = idx_start

    peak_loc = np.full(len(peak_loc1), np.nan)
    for n in range(len(peak_loc1)):
        start = peak_loc1[n] - int(np.ceil(L / 2)) + 1
        end = peak_loc1[n] + int(np.ceil(L / 2))
        temp = signal[start:end]
        temp_idx = np.argmax(temp)
        peak_loc[n] = temp_idx + start - 1

    peak_loc = np.unique(peak_loc[~np.isnan(peak_loc)]).astype(int)
    return peak_loc
