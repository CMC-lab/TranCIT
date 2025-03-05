import numpy as np
from scipy.spatial.distance import euclidean


def find_best_shrinked_locs(D, shrinked_locs, all_locs):
    
    Nfull, _ = np.histogram(D[all_locs], bins=100, density=True)

    distance = np.full(len(shrinked_locs), np.nan)
    
    for n in range(100, len(shrinked_locs)):
        Ntemp, _ = np.histogram(D[shrinked_locs[:n]], bins=100, density=True)
        distance[n] = euclidean(Nfull, Ntemp)

    best_n = np.nanargmin(distance)
    best_locs = shrinked_locs[:best_n]

    return best_locs, distance


def shrink_locs_resample_uniform(loc, L):
    Ngen = 0
    maxNgen = len(loc)
    shrinked_locs = np.full(maxNgen, np.nan)
    loc_range = np.copy(loc)

    while Ngen < maxNgen:
        if loc_range.size == 0:
            break
        selected_loc = loc_range[np.random.randint(0, len(loc_range))]
        shrinked_locs[Ngen] = selected_loc
        loc_range = loc_range[np.abs(selected_loc - loc_range) >= L]

        Ngen += 1

    shrinked_locs = shrinked_locs[~np.isnan(shrinked_locs)]

    return shrinked_locs
