import numpy as np

def extract_events(A, cumP, L_start, L):
    A_event = np.full((L, len(cumP)), np.nan)
    
    for i in range(len(cumP)):
        idx = np.arange(cumP[i] - L_start + 1, cumP[i] + L - L_start + 1)
        A_event[:, i] = A[idx]
    
    return A_event
