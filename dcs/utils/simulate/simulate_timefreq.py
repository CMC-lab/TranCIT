import numpy as np

def get_simul_timefreq(Yt_event_mc, Yt_stats, spectr_params):
    _, L, ntrials = Yt_event_mc.shape
    nvar = spectr_params['simobj']['nvar']
    N_seg = spectr_params['N_seg']
    
    # Initialize the PSD array with NaNs
    Yt_stats['spectr']['psd'] = np.full((nvar, spectr_params['nfreqs'], L, ntrials), np.nan)
    
    # Compute the time-frequency representation for each variable
    for n in range(nvar):
        for i in range(ntrials // N_seg):
            idx = slice(N_seg * i, N_seg * (i + 1))
            Yt_stats['spectr']['psd'][n, :, :, idx], freqs_cond1, timesout_cond1 = timefreqMB(
                Yt_event_mc[n, :, idx], 
                spectr_params['fs'], 
                freqs=spectr_params['freqs'], 
                nfreqs=spectr_params['nfreqs']
            )
        
        # Handle any remaining trials
        if N_seg * (i + 1) < ntrials:
            idx = slice(N_seg * (i + 1), ntrials)
            Yt_stats['spectr']['psd'][n, :, :, idx], freqs_cond1, timesout_cond1 = timefreqMB(
                Yt_event_mc[n, :, idx], 
                spectr_params['fs'], 
                freqs=spectr_params['freqs'], 
                nfreqs=spectr_params['nfreqs']
            )
    
    # Store frequency and time information
    Yt_stats['spectr']['time'] = timesout_cond1
    Yt_stats['spectr']['freq'] = freqs_cond1
    
    # Apply averaging if specified
    if spectr_params['average_flag']:
        Yt_stats['spectr']['psd'] = np.mean(np.abs(Yt_stats['spectr']['psd']), axis=3)
    
    return Yt_stats
