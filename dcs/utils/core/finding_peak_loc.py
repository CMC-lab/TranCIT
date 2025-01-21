import numpy as np

def find_peak_loc(signal, loc, L):
    idx_start = 0
    idx_end = 0
    
    loc = loc[(loc >= L) & (loc <= len(signal) - L)]

    peak_loc1 = []
    
    while idx_end < len(loc) - 1:
        while loc[idx_end + 1] - loc[idx_start] < L:
            idx_end += 1
            if idx_end == len(loc) - 1:
                break

        temp = range(idx_start, idx_end + 1)

        temp_signal = signal[loc[temp]]
        temp_idx = np.argmax(temp_signal)
        peak_loc1.append(loc[temp_idx + idx_start])

        idx_start = idx_end + 1
        idx_end = idx_start

    peak_loc = np.full(len(peak_loc1), np.nan)
    for n in range(len(peak_loc1)):
        temp = signal[peak_loc1[n] - (L // 2) + 1 : peak_loc1[n] + (L // 2)]
        temp_idx = np.argmax(temp)
        peak_loc[n] = temp_idx + peak_loc1[n] - (L // 2)

    peak_loc = np.unique(peak_loc)
    return peak_loc
