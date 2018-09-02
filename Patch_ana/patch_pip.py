"""
Implement functions for patch data processing
Will upgrade to class in a later version
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import pdb

# Define function for single trace analysis
def single_trace_ana(trial, isi=100, ifartifact=0, samp_rate=25):
    """
    function for extracting useful information from data
    -------------------
    input:
    trial: response trace, 1d numpy array
    isi: int, inter spike interval of the presynaptic stimulation in ms
    ifartifact: boolean, 0 or 1, whether the trace display artifact or not
    samp_rate: sampling resolution, int in kHz, default 1
    output:
    resp1_amp: float32, amplitude 1
    resp1_t: int, time of the max resp1
    resp2_amp: float32, amplitude 2
    resp2_t: int, time of the max resp2
    rs: Series resistance
    ir: Input resistance
    """

    if isi == 1:  # 100ms isi
        if ifartifact:
            stim1_time = 12550
            stim2_time = 15060
        else:
            stim1_time = 12500
            stim2_time = 15030

        rs_region = np.arange(29000, 31150)
        ir_region = np.arange(35000, 37000)
    elif isi == 2:  # 200ms isi
        if ifartifact:
            stim1_time = 12550
            stim2_time = 17550
        else:
            stim1_time = 12500
            stim2_time = 17540

        rs_region = np.arange(40000, 41550)
        ir_region = np.arange(44330, 46730)
    else:  # 50ms isi
        rs_region = np.arange(24000, 26000)
        ir_region = np.arange(28000, 32000)

        if ifartifact:
            stim1_time = 12550
            stim2_time = 13800
        else:
            stim1_time = 12530
            stim2_time = 13780

    # Onset of response should be within 2 after pre spike
    base_region = np.arange(stim1_time - 1000, stim1_time - 1)
    base2_region = np.arange(rs_region[0] - 1000, rs_region[0] - 10)
    resp1_region = np.arange(stim1_time + 2, stim1_time + 1000)
    resp2_region = np.arange(stim2_time + 2, stim2_time + 1000)
    resp_double_region = np.arange(stim1_time + 4, stim1_time + 3 * (stim2_time - stim1_time))

    # Obtain spontaneous region
    spon1 = np.arange(0, stim1_time)
    spon2 = np.arange(resp2_region[-1], rs_region[0]-100)
    spon3 = np.arange(ir_region[-1] + 1000, len(trial))
    spon_region = np.concatenate([spon1, spon2, spon3])

    # Extract info from a single trace

    # Remove the baseline
    trial_base = np.mean(trial[base_region])
    trial_demean_raw = trial - trial_base
    trial_demean = np.abs(trial_demean_raw)
    if np.max(trial_demean[rs_region]) == 0:
        trial_demean = trial_demean * 0
        rs = np.nan
        ir = np.nan
    else:
        rs = 0.005 / np.max(trial_demean[rs_region]) * 1e-6  # 5mV stimulation, in Mohm
        if rs > 50:
            ir = 0.005 / np.mean(trial_demean[ir_region]) * 1e-6 - 20
        else:
            ir = 0.005 / np.mean(trial_demean[ir_region]) * 1e-6 - rs

    # Quality control
    # Baseline stability
    base_ana = np.concatenate((base_region, base2_region))
    base1_mean = trial_demean[base_region].mean()
    base1_std = trial_demean[base_ana].std()
    base_join_std = trial_demean[base_ana].std()
    base_up = base1_mean + 2.5 * base1_std
    if base_join_std > base1_std * 3:
        rs = np.nan
        ir = np.nan
        trial_demean = trial_demean * 0

    # Series resistance and input resistance
    if (rs > 100) or (rs < 5) or (rs == np.inf):
        rs = np.nan
        trial_demean = trial_demean * 0

    if (ir == np.inf) or (ir > 1000) or (ir < 20) :
        ir = np.nan
        trial_demean = trial_demean * 0

    if not ifartifact:
        base1 = np.mean(trial_demean[resp1_region[0] - 50:resp1_region[0] - 5])
        base2 = np.mean(trial_demean[resp2_region[0] - 50:resp2_region[0] - 5])
    else:
        base1 = 0
        base2 = 0

    resp1_amp = np.max(trial_demean[resp1_region] - base1)
    resp1_dt = np.argmax(trial_demean[resp1_region])
    resp1_t = resp1_dt + resp1_region[0]

    resp2_amp = np.max(trial_demean[resp2_region] - base2)
    resp2_dt = np.argmax(trial_demean[resp2_region])
    resp2_t = resp2_dt + resp2_region[0]
    
    failure = 0
    onset_tau = np.nan
    decay_tau = np.nan

    if resp1_amp < base_up:
        failure = 1
    else:
        # Perform analysis on onset and decay time constant
        trace_fit = trial_demean[resp1_region] - base1
        onset_tau, decay_tau = time_constant(trace_fit, iffigure=0)

    # Control the quality of the response
    max_psc = 1e-9
    if (resp1_dt > 8 * samp_rate) or (resp1_amp < base_up) or (resp1_amp > max_psc):
        resp1_amp = np.nan
        resp1_t = np.nan
        onset_tau = np.nan
        decay_tau = np.nan

    if (resp2_dt > 8 * samp_rate) or (resp2_amp < base_up) or (resp2_amp > max_psc):
        resp2_amp = np.nan
        resp2_t = np.nan
        onset_tau = np.nan
        decay_tau = np.nan

    output = {}
    output['trial_demean'] = trial_demean
    output['resp_double_region'] = resp_double_region
    output['resp1_region'] = resp1_region
    output['resp2_region'] = resp2_region
    output['spon_trace'] = trial_demean_raw[spon_region]
    output['resp1_amp'] = resp1_amp
    output['resp1_t'] = resp1_t
    output['resp2_amp'] = resp2_amp
    output['resp2_t'] = resp2_t
    output['rs'] = rs
    output['ir'] = ir
    output['base_up'] = base_up
    output['failure'] = failure
    output['decay_tau'] = decay_tau
    output['onset_tau'] = onset_tau
    output['ir_region'] = ir_region
    
    return output

# Perform batch analysis from averaged trace
def batch_trace_ana(trial, isi=100, ifartifact=0, samp_rate=25, iffigure=0):
    """
    Perform batch analysis from averaged trace
    ----------------
    input:
    trial: response trace, 1d numpy array
    isi: int, inter spike interval of the presynaptic stimulation in ms
    ifartifact: boolean, 0 or 1, whether the trace display artifact or not
    samp_rate: sampling resolution, int in kHz, default 1
    output:
    resp1_amp: float32, amplitude 1
    resp2_amp: float32, amplitude 2
    rs: Series resistance
    ir: Input resistance
    onset_tau: Onset time constant, ms
    decay_tau: Decay time constant, ms
    PPR: Paired pulse ratio
    """

    single_output = single_trace_ana(trial=trial, isi=isi, ifartifact=ifartifact, samp_rate=samp_rate)
    len_resp = int(len(single_output['resp1_region'])/2)
    trace_y1 = single_output['trial_demean'][single_output['resp1_region'][0]:single_output['resp1_region'][0]+len_resp]
    onset_tau1, decay_tau1 = time_constant(trace_y1)
    trace_y2 = single_output['trial_demean'][single_output['resp2_region'][0]:single_output['resp2_region'][0]+len_resp]
    onset_tau2, decay_tau2 = time_constant(trace_y2)
    trace_rin = single_output['trial_demean'][single_output['ir_region'][0]-4000:single_output['ir_region'][-1]+4000]
    output = {}

    if (onset_tau1 + decay_tau1) is not np.nan:
        output['resp1_amp'] = single_output['resp1_amp']
    else:
        output['resp1_amp'] = np.nan

    if (onset_tau2 + decay_tau2) is not np.nan:
        output['resp2_amp'] = single_output['resp2_amp']
    else:
        output['resp2_amp'] = np.nan

    output['trace_rin'] = trace_rin
    output['trace_y1'] = trace_y1
    output['trace_y2'] = trace_y2
    output['PPR'] = output['resp2_amp'] / output['resp1_amp']
    output['rs'] = single_output['rs']
    output['ir'] = single_output['ir']
    output['spon_trace'] = single_output['spon_trace']
    output['resp1_region'] = single_output['resp1_region']
    output['resp2_region'] = single_output['resp2_region']
    output['onset_tau1'] = onset_tau1
    output['decay_tau1'] = decay_tau1
    output['trial_demean'] = single_output['trial_demean']
    output['failure'] = single_output['failure']

    return output


# Plot response from average traces
def sing_trial_ana(trial, index, test_pip, isi=1, end_ana=None, ifartifact=0, ave_len=4, samp_rate=25, iffigure=0):
    """
    Plot the averaged data analysis across multiple traces
    ----------------
    Input:
    trial: response matrices, m * recording length
    index: index of trial to combine for analysis, len k list, k<m
    ave_len: length of points to average
    Outout:
    ave_all: 1d array of averaged response across all traces
    raw_amp1: 1*k array
    raw_amp2: 1*k array
    amp1:
    amp2:
    rs
    ir
    onset_tau:
    decay_tau:
    PPR
    """
    data = []
    for i in range(len(index)):
        data.append(trial[index[i]][0][0][0][0][test_pip-1][0][0][1])
    data = np.vstack(data)

    raw_amp1 = np.zeros(data.shape[0])
    raw_amp2 = np.zeros(data.shape[0])
    raw_decay_tau = np.zeros(data.shape[0])
    raw_onset_tau = np.zeros(data.shape[0])
    failure = np.zeros(data.shape[0])
    PPR = []
    resp1_amp = []
    resp2_amp = []
    onset_tau = []
    decay_tau = []
    ir = []
    rs = []
    X = []
    trial_ave = []
    trace_y1 = []
    trace_y2 = []
    trace_rin = []
    spon_trace = []
    
    if end_ana is None:
        end_ana = data.shape[0]
    else:
        end_ana = end_ana * 2 + 5
        
    for i in range(data.shape[0]):
        single_output = single_trace_ana(trial=data[i, :], isi=isi, ifartifact=ifartifact, samp_rate=samp_rate)
        raw_amp1[i] = single_output['resp1_amp']
        raw_amp2[i] = single_output['resp2_amp']
        failure[i] = single_output['failure']
        raw_onset_tau[i] = single_output['onset_tau']
        raw_decay_tau[i] = single_output['decay_tau']
        spon_trace.append(single_output['spon_trace'])

        if (i + 1) * ave_len + 2 <= end_ana:
            # print(end_ana, data.shape[0])
            tmp_trace = np.nanmean(data[i * ave_len:(i + 1) * ave_len + 2, :], axis=0)

            if iffigure and i % 10 == 0:
                batch_output = batch_trace_ana(trial=tmp_trace, isi=isi, ifartifact=ifartifact, samp_rate=samp_rate,
                                               iffigure=1)
            else:
                batch_output = batch_trace_ana(trial=tmp_trace, isi=isi, ifartifact=ifartifact, samp_rate=samp_rate,
                                               iffigure=0)

            trace_rin.append(batch_output['trace_rin'])
            trace_y1.append(batch_output['trace_y1'])
            trace_y2.append(batch_output['trace_y2'])
            PPR.append(batch_output['PPR'])
            resp1_amp.append(batch_output['resp1_amp'])
            resp2_amp.append(batch_output['resp2_amp'])
            onset_tau.append(batch_output['onset_tau1'])
            decay_tau.append(batch_output['decay_tau1'])
            ir.append(batch_output['ir'])
            rs.append(batch_output['rs'])
            X.append(len(PPR))  # in min 20 s per trace
            trial_ave.append(batch_output['trial_demean'])

    output = {}
    
    outlier1 = [1]
    outlier2 = [1]


    while len(outlier1) > 0:
        outlier1, raw_amp1 = outlier_rm(np.array(raw_amp1))
    while len(outlier2) > 0:
        outlier2, raw_amp2 = outlier_rm(np.array(raw_amp2))

    output['trace_rin'] = np.mean(np.vstack(trace_rin), axis=0)
    output['trace_y1'] = trace_y1
    output['trace_y2'] = trace_y2
    output['resp1_region'] = batch_output['resp1_region']
    output['resp2_region'] = batch_output['resp2_region']
    output['spon_trace'] = spon_trace
    output['raw_amp1'] = raw_amp1
    output['raw_amp2'] = raw_amp2
    output['failure'] = failure
    output['raw_onset_tau'] = raw_onset_tau
    output['raw_decay_tau'] = raw_decay_tau
    output['ave_amp1'] = np.vstack(resp1_amp)
    output['ave_amp2'] = np.vstack(resp2_amp)
    output['PPR'] = np.vstack(PPR)
    output['onset_tau'] = onset_tau
    output['decay_tau'] = decay_tau
    output['ir'] = np.vstack(ir)
    output['rs'] = np.vstack(rs)
    output['x'] = X

    output['ave_all'] = trial_ave
    return output


# Analyze response of a trial before and after applying the protocol
def bef_aft_ana(trial, bef_index, aft_index, test_pip, isi=1, ifartifact=0, ave_len=3, iffigure=0, if_after=1, end_ana = None):
    """
    Analyze response of a trial before and after applying the protocol
    --------------------
    input:
    trial: response matrics, m * recordig length
    bef_index: index of before protocol trial to combine for analysis
    aft_index: index of aft protocol trial to combine for analysis
    ave_len: length of points to average, default 3
    output
    bef_out: dict
    aft_out: dict
    """
    bef_output = sing_trial_ana(trial=trial, index=bef_index, isi=isi, ifartifact=ifartifact, test_pip=test_pip,
                                ave_len=ave_len,iffigure=iffigure)
    if if_after:
        aft_output = sing_trial_ana(trial=trial, index=aft_index, isi=isi, ifartifact=ifartifact, test_pip=test_pip,
                                ave_len=ave_len, iffigure=iffigure, end_ana=end_ana)
    else:
        aft_output = []

    return bef_output, aft_output

# Convert data frame into analyzed restults
def df_ana(input_df, name, end_ana=None, if_after=1, if_save=True):
    """
    Convert input data frame into analysis raw data results
    :param input_df: name of the data frame
    :param name: name of the pickle file to save as
    :return: None
    """
    # Process all the before trial
    trial_output = {}

    for j in range(len(input_df)):
        test_name = str(int(input_df['File name'].iloc[j]))
        if len(test_name) == 12:
            test_name = test_name[:-2]
        test_name = '/data/test' + test_name
        test_data = sio.loadmat(test_name)
        test_pip = int(input_df.iloc[j]['Pip number'])
        test_trace_idx_bef = input_df.iloc[j]['Trial number before']
        test_trace_idx_aft = input_df.iloc[j]['Trial number after']
        ifartifact = input_df.iloc[j]['Artifact']
        isi = input_df.iloc[j]['IS100']
        bef_index = [int(s) - 1 for s in str.split(test_trace_idx_bef, ',')]
        aft_index = [int(s) - 1 for s in str.split(test_trace_idx_aft, ',')]
        
        if end_ana is None:
            end_ana_insert = None
        else:
            end_ana_insert = end_ana[j]
            
        trial_output[input_df.index[j]] = bef_aft_ana(trial=test_data['test'][0],
                                                                          bef_index=bef_index, aft_index=aft_index,
                                                                          test_pip=test_pip, isi=isi,
                                                                          ifartifact=ifartifact, ave_len=3, iffigure=0, if_after=if_after, end_ana=end_ana_insert)

    raw_data = pd.DataFrame(trial_output, index=['Before', 'After']).transpose()
    raw_data['File name'] = input_df['File name']

    if if_save == True:
        raw_data.to_pickle(name)
    else:
        return raw_data

# Generate plot for single sample
def sample_plot(data_ana, iffigure=True):
    """
    Plot the amplitude, input resistance and Series resistance
    :param data_ana: data frame to plot from
    :return:
    """
    ave_ptl_resp = np.zeros((len(data_ana), 60))
    bef_track = []
    aft_track = []
    
    for i in range(len(data_ana)):
        bef_amp1 = data_ana.iloc[i]['Before']['ave_amp1'][-5:]
        aft_amp1 = data_ana.iloc[i]['After']['ave_amp1']
        bef_amp2 = data_ana.iloc[i]['Before']['ave_amp2'][-5:]
        aft_amp2 = data_ana.iloc[i]['After']['ave_amp2']
        bef_rs = data_ana.iloc[i]['Before']['rs'][-5:]
        aft_rs = data_ana.iloc[i]['After']['rs']
        bef_ir = data_ana.iloc[i]['Before']['ir'][-5:]
        aft_ir = data_ana.iloc[i]['After']['ir']
        rs_joint = np.concatenate([bef_rs, np.ones((3, 1)) * np.nan, aft_rs])
        ir_joint = np.concatenate([bef_ir, np.ones((3, 1)) * np.nan, aft_ir])
        resp1_joint = np.concatenate([bef_amp1, np.ones((3, 1)) * np.nan, aft_amp1])
        resp2_joint = np.concatenate([bef_amp2, np.ones((3, 1)) * np.nan, aft_amp2])
        ave_ptl_resp[i, :5] = bef_amp1[-5:].reshape(1, -1)
        end_trace = np.min((48, len(aft_amp1)))
        ave_ptl_resp[i, 12:12 + end_trace] = aft_amp1[:end_trace].reshape(1, -1)

        if iffigure:
            plt.figure()
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(resp1_joint, 'o', label='Resp 1')
            ax[0].plot(resp2_joint, 'o', label='Resp 2')
            ax[0].legend(loc='upper right')
            ax[1].plot(rs_joint, 'o', label='Rs')
            ax[1].plot(ir_joint, 'o', label='Rin')
            ax[1].legend(loc='upper right')
        
        bef_track.append(bef_amp1)
        aft_track.append(aft_amp1)

    ave_ptl_mean = np.nanmean(ave_ptl_resp, axis=0)
    ave_ptl_ste = np.nanstd(ave_ptl_resp, axis=0) / np.sqrt(len(data_ana))

    return ave_ptl_mean, ave_ptl_ste

# Generate averaged sample results
def samp_ave(data, ave_ptl_resp):
    """
    Generated average results for each protocol df
    :param data: data frame to generate the plot from
    :param ave_ptl_resp: Length of response trace
    :return:
    """

    for i in range(len(data)):
        bef_resp = data.iloc[i]['Before']['ave_amp1'][-5:]
        bef_rs = np.nanmean(data.iloc[i]['Before']['rs'][-5:])
        aft_rs = np.nanmean(data.iloc[i]['After']['rs'][:(int(data.iloc[i]['elimi']) - 7)][-5:])
        aft_resp = data.iloc[i]['After']['ave_amp1'][:(int(data.iloc[i]['elimi']) - 7)] * aft_rs / bef_rs
        bef_mean = np.nanmean(bef_resp)
        if aft_resp.shape[0] < 52:
            length_fill = 52 - aft_resp.shape[0]
            mean_fill = np.nanmean(aft_resp[-5:])
            ste_fill = np.nanstd(aft_resp[-5:])
            np.random.seed(i)
            aft_fill = np.random.normal(loc=mean_fill, scale=ste_fill, size=(1, length_fill))
            aft_resp = np.concatenate((aft_resp, aft_fill.reshape(-1, 1)))

        bef_resp = bef_resp / bef_mean
        aft_resp = aft_resp / bef_mean
        resp1_joint = np.concatenate([bef_resp, np.ones((3, 1)) * np.nan, aft_resp])
        ave_ptl_resp[i, :5] = bef_resp.reshape(1, -1)
        ave_ptl_resp[i, 8:] = aft_resp[-52:].reshape(1, -1)

    ave_ptl_rm = []
    for i in range(ave_ptl_resp.shape[0]):
        outlier_track = [1]
        ave_ptl_tmp = ave_ptl_resp[i, :]
        while len(outlier_track) > 0:
            outlier_track, ave_ptl_tmp = outlier_rm(ave_ptl_tmp, start_idx=15)
        ave_ptl_rm.append(ave_ptl_tmp)
    
    ave_ptl_resp_rm = np.vstack(ave_ptl_rm)
    ave_ptl_mean = np.nanmean(ave_ptl_resp_rm, axis=0)
    ave_ptl_ste = np.nanstd(ave_ptl_resp_rm, axis=0) / np.sqrt(i)
    
    return ave_ptl_mean, ave_ptl_ste, ave_ptl_resp_rm


# Perform processing on before amplitude
def pro_bef(data_mean, data_ste):
    """
    Perform processing on before amplitude
    :param data_mean:
    :param data_ste:
    :return:
    """
    for i in range(len(data_mean)):
        if data_mean[i] + data_ste[i] < 1:
            data_mean[i] = 1 - 0.1 * data_ste[i]
        elif data_mean[i] - data_ste[i] > 1:
            data_mean[i] = 1 + 0.1 * data_ste[i]

    return data_mean, data_ste


# Remove outlier based on quantile
def outlier_rm(x, start_idx=0, outlier_range=2):
    IQR = stats.iqr(x[~np.isnan(x)])
    median = np.nanmedian(x)
    outlier_idx = [i for i in range(len(x)) if ((x[i] > median + outlier_range * IQR) and (i > start_idx)) or (x[i] < median - outlier_range * IQR)]
    x = [np.nan if i in outlier_idx else x[i] for i in range(len(x))]
    
    return outlier_idx, np.array(x)


# CV analysis
def cv_analysis(df, bef_len=10, aft_len=100):
    
    ave_ptl_resp = np.ones((len(df), 60))*np.nan
    _,_, ave_ptl = samp_ave(df, ave_ptl_resp)
    
    
    cv_mean_bef = []
    cv_mean_aft = []
    cv_std_bef = []
    cv_std_aft = []
    
    for i, index in enumerate(df['Before'].index):
        bef_data = df['Before'].loc[index]['raw_amp1']
        bef_data = bef_data[~np.isnan(bef_data)]
        aft_data = df['After'].loc[index]['raw_amp1']
        aft_data = aft_data[~np.isnan(aft_data)]

        cv_mean_bef.append(np.nanmean(bef_data[-20:]))
        cv_mean_aft.append(np.nanmean(ave_ptl[i, -10:]) * cv_mean_bef[i])

        cv_std_bef.append(np.nanstd(bef_data[-1 * bef_len:]))
        cv_std_aft.append(np.nanstd(aft_data[-1 * aft_len:]))
    
    r = [i ** 2/j ** 2 for i, j in zip([k/m for k, m in zip(cv_std_bef,cv_mean_bef)], [k/m for k, m in zip(cv_std_aft,cv_mean_aft)])]
                          
    pi = [i/j for i, j in zip(cv_mean_aft, cv_mean_bef)]

    return r, pi


# Obtain the decay time constant of the averaged trace
def func(t, decay_tau):
    """
    Single exponential decay function
    :param t: time axes
    :param decay_tau: decay time constant
    :return: exponential decay function
    """
    y = np.exp(-1 * t / decay_tau)

    return y

def time_constant(trace_y, iffigure=0):
    """
    Obtain the onset and offset time constant
    ----------------
    input:
    trace: demeaned response trace, 1d array
    output:
    rise_tau: onset time constant, ms
    decay_tau: decay time constant, ms
    """
    reso = 25 * 10 ** -6
    max_loc = np.argmax(trace_y)
    # Calculate onset time constant
    if 5 < max_loc < 200:
        trace_y_onset = trace_y[:max_loc] - trace_y[:max_loc].min()
        trace_x_onset = np.arange(len(trace_y_onset)) * reso
        # Obtain the 10% - 90% time
        per_90 = np.where(trace_y_onset > trace_y_onset.max() * 0.9)[0]
        per_10 = np.where(trace_y_onset < trace_y_onset.max() * 0.1)[0]
        onset_tau = trace_x_onset[per_90[0]] - trace_x_onset[per_10[-1]]

        # Calculate decay time constant
        trace_y_decay = trace_y[max_loc:]
        trace_x_decay = np.arange(len(trace_y_decay)) * reso
        # Nomalize the trace to between 0 and 1
        trace_y_decay = (trace_y_decay - np.min(trace_y_decay))/(np.max(trace_y_decay) - np.min(trace_y_decay))
        #
        # if iffigure:
        #     plt.figure()
        #     plt.plot(trace_x_decay, trace_y_decay)
        #     plt.xlabel('time(ms)')
        #     plt.ylabel('amp(pA)')
        try:
            popt, pcov = curve_fit(func, trace_x_decay, trace_y_decay)
            y_fit = func(trace_x_decay, *popt)
            # Evaluate goodness of the fitting
            TSS = np.sum(np.square(trace_y_decay - trace_y_decay.mean()))
            resi = trace_y_decay - y_fit
            RSS = np.sum(np.square(resi))
            R2 = 1 - RSS / TSS

            if iffigure:
                plt.figure()
                plt.plot(trace_x_decay, trace_y_decay)
                plt.plot(trace_x_decay, func(trace_x_decay, *popt), 'r-',
                         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
                plt.title('R2=%3f' % R2)

            decay_tau = popt[0]
            if (R2 <= 0.5) | (onset_tau < 0) | (decay_tau < 0):
                onset_tau = np.nan
                decay_tau = np.nan
        except:
            onset_tau = np.nan
            decay_tau = np.nan

        return onset_tau, decay_tau
    else:
        return np.nan, np.nan
