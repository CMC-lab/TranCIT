import numpy as np


def remove_artif_trials(Yt_events, locs, lower_ths):
    idxall = np.where(Yt_events[:2, :, :] < lower_ths)
    
    itrial_remove = np.unique(idxall[2])
    
    Yt_events = np.delete(Yt_events, itrial_remove, axis=2)
    locs = np.delete(locs, itrial_remove)
    
    print(f'removed: {len(itrial_remove)} artifact trials')
    
    return Yt_events, locs
