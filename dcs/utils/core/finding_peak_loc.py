import numpy as np

def find_peak_loc(signal, loc, L):
    loc = loc[(loc >= L) & (loc <= len(signal) - L)] # loc[loc <= len(signal) - L - 1]

    peak_loc1 = []
    idx_start = 0
    idx_end = 0
    
    while idx_end < len(loc):
        while loc[idx_end + 1] - loc[idx_start] < L:
            idx_end += 1
            if idx_end == len(loc) - 1:
                break

        temp = np.arange(idx_start, idx_end + 1)
        temp_signal = signal[loc[temp]]
        temp_idx = np.argmax(temp_signal)
        peak_loc1.append(loc[temp_idx + idx_start])

        idx_start = idx_end + 1
        idx_end = idx_start

    peak_loc = np.full(len(peak_loc1), np.nan)
    for n in range(len(peak_loc1)):
        start = peak_loc1[n] - int(np.ceil(L / 2))
        end = peak_loc1[n] + int(np.ceil(L / 2)) + 1
        temp = signal[start:end]
        temp_idx = np.argmax(temp)
        peak_loc[n] = temp_idx + start

    peak_loc = np.unique(peak_loc[~np.isnan(peak_loc)]).astype(int)
    return peak_loc
