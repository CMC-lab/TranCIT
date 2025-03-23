import numpy as np
from dcs.causality.time_varying import time_varying_causality
from dcs.src.bic import multi_trial_BIC
from dcs.utils.core.finding_best_shrinked_locs import (
    find_best_shrinked_locs, shrink_locs_resample_uniform)
from dcs.utils.core.finding_peak_loc import find_peak_loc
from dcs.utils.core.getting_Yt import get_Yt, get_Yt_stats
from dcs.utils.core.residuals import get_residuals
from dcs.utils.preprocessing.removing_artif_trials import remove_artif_trials
from dcs.utils.simulate.ar import simul_AR_event, simul_AR_event_btsp
from dcs.utils.simulate.timefreq import get_simul_timefreq


def snapshot_detect_analysis_pipeline(OriSignal, DetSignal, Params):
    SnapAnalyOutput = {}

    if Params['Options']['Detection'] == 1:
        # SET THRESHOLD
        D = DetSignal[0]
        d0 = Params['Detection'].get('d0', np.nanmean(D) + Params['Detection']['ThresRatio'] * np.nanstd(D))
                
        # FIND REFERENCE POINTS
        temp_loc = np.where(D >= d0)[0]
        if Params['Detection']['AlignType'] == "peak":
            # align either on detection signal or original 
            locs = find_peak_loc(DetSignal[1], temp_loc, Params['Detection']['L_extract'])
        elif Params['Detection']['AlignType'] == "pooled":
            if Params['Detection']['ShrinkFlag']:
                locs = shrink_locs_resample_uniform(temp_loc, int(np.ceil(Params['Detection']['L_extract'] / 2)))
                locs, ـ = find_best_shrinked_locs(D, locs, temp_loc)
            else:
                locs = temp_loc
    else:
        # PRE-DEFINED REFERENCE POINTS
        locs = Params['Detection']['locs']

    # REMOVE BORDER POINTS
    locs = locs[(locs >= Params['Detection']['L_extract']) & (locs <= OriSignal.shape[1] - Params['Detection']['L_extract'])]

    # BIC MODEL ESTIMATION
    if Params['Options']['BIC']:
        print('Performing BIC model selection')
        BICParser = {
            'OriSignal': OriSignal,
            'DetSignal': DetSignal,
            'Params': Params,
            'EstimMode': 'OLS'
        }
        if Params['BIC']['mode'] == 'biased':
            Yt_events_momax = get_Yt(OriSignal,
                                     locs,
                                     Params['BIC']['momax'],
                                     Params['BIC']['tau'],
                                     Params['Detection']['L_start'],
                                     Params['Detection']['L_extract'])
            BICoutputs = multi_trial_BIC(Yt_events_momax, BICParser) # For empirical data
            morder = BICoutputs['mobic'][1]
        np.savez(f"{Params['Output']['FileKeyword']}_BIC.npz", Params=Params, BICoutputs=BICoutputs)
    else:
        # preassigned model order
        morder = Params['BIC']['morder']
        BICoutputs = None

    # EXTRACT EVENT SNAPSHOTS
    Yt_events = get_Yt(OriSignal,
                       locs,
                       morder,
                       Params['BIC']['tau'], 
                       Params['Detection']['L_start'],
                       Params['Detection']['L_extract'])

    if Params['Detection']['remove_artif']:
        Yt_events, locs = remove_artif_trials(Yt_events, locs, -15000)

    # CALCULATE D-DEPENDENT SNAPSHOT STATISTICS
    Yt_stats = get_Yt_stats(Yt_events, morder)

    # CAUSAL ANALYSIS
    if Params["Options"]["CausalAnalysis"]:
        CausalParams = Params['CausalParams']
        CausalParams['morder'] = morder  # Set model order
        CausalOutput = {}
        CausalOutput['OLS'] = time_varying_causality(Yt_events, Yt_stats, CausalParams)
    
    # perform bootstrapping
    if Params['Options']['Bootstrap']:
        print('Start Bootstrapping!')
        Params['MonteC_Params']['morder'] = morder
        # Params['MonteC_Params']['Ntrials'] = len(locs)
        Et = get_residuals(Yt_events, Yt_stats)
        
        for n_btsp in range(1, Params['MonteC_Params']['Nbtsp'] + 1):
            print(f'Calculating bootstrap trial: {n_btsp}')
            
            Yt_events_btsp = simul_AR_event_btsp(Params['MonteC_Params'], Yt_events, Yt_stats, Et) 
            Yt_stats_btsp = get_Yt_stats(Yt_events_btsp, morder)
            
            CausalOutput_btsp = {}
            CausalOutput_btsp['OLS'] = time_varying_causality(Yt_events_btsp, Yt_stats_btsp, CausalParams)
            
            file_keyword = Params['Output']['FileKeyword']
            output_filename = f"{file_keyword}_btsp_{n_btsp}_model_causality.npz"
            np.savez_compressed(output_filename, Params=Params, CausalOutput_btsp=CausalOutput_btsp, Yt_stats_btsp=Yt_stats_btsp)

    # calculate power spectral density
    if Params['Options']['PSD']:
        if not Params['PSD']['MonteC_flag']:
            Yt_stats = get_simul_timefreq(Yt_events, Yt_stats, Params['PSD'])
        else:
            Params['PSD']['simobj']['morder'] = morder
            Yt_events_mc = simul_AR_event(Params['PSD']['simobj'], Yt_stats)
            Yt_stats = get_simul_timefreq(Yt_events_mc, Yt_stats, Params['PSD'])
    
    # SAVE RESULTS    
    SnapAnalyOutput["d0"] = d0 if Params["Options"]["Detection"] else None
    SnapAnalyOutput['locs'] = locs
    SnapAnalyOutput['morder'] = morder
    SnapAnalyOutput['Yt_stats'] = Yt_stats
    SnapAnalyOutput['CausalOutput'] = CausalOutput if Params["Options"]["CausalAnalysis"] else None
    SnapAnalyOutput['BICoutputs']   = BICoutputs if Params['Options']['BIC'] else None

    # OUTPUT
    if Params['Options']['save_flag']:
        Yt_stats['mean'] = Yt_stats['mean'][:2, :]
        Yt_stats['Sigma'] = Yt_stats['Sigma'][:, :2, :2]
        Params['DeSnap_inputs'] = {'x': [],
                                   'y': OriSignal,
                                   'yf': DetSignal,
                                   'D': SnapAnalyOutput["d0"],
                                   'Yt_stats_cond': Yt_stats}

        file_keyword = Params['Output']['FileKeyword']
        if Params['Options']['PSD']:
            PSD = Yt_stats['spectr']
            np.savez_compressed(f"{file_keyword}_psd.npz", 
                                Params=Params, 
                                PSD=PSD)
        else:
            if Params["Options"]["CausalAnalysis"]:
                np.savez_compressed(f"{file_keyword}_model_causality.npz", 
                                    Params=Params, 
                                    Yt_stats=Yt_stats, 
                                    CausalOutput=CausalOutput,
                                    SnapAnalyOutput=SnapAnalyOutput)
            else:
                np.savez_compressed(f"{file_keyword}_model.npz", 
                                    Params=Params, 
                                    Yt_stats=Yt_stats)

    return SnapAnalyOutput, Params, Yt_events
