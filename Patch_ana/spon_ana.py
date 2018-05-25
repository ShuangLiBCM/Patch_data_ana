"""
    Analyse spontaneous EPSC, IPSC for identifying plasticity effect
"""

import numpy as np
import scipy as sc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle

def pop_Spon_Trace(trace_list, elimi_list):
    """
    Choose uncontaminated spontaneous firing trace
    :param trace_list:
    :param elimi_list:
    :return: trace_list: cleaned list of spontaneous firing traces
    """
    while len(elimi_list) > 0:
        trace_list.pop(elimi_list[-1])
        elimi_list.pop(-1)

    return trace_list

def fun_exp(t, decay_tau):
    """
    Single exponential decay function
    :param t: time axes
    :param decay_tau: decay time constant
    :return: exponential decay function
    """
    y = np.exp(-1 * t / decay_tau)
    return y

def fun_2exp(t, a, tau1, tau2):
    """
    Double exponential decay function
    :param t: time axes
    :param a: relative amplitude for fast and slow decay component
    :param tau1: fast decay time constant
    :param tau2: slow decay time constant
    :return: Double exponential decay function
    """
    y = a * np.exp(-1 * t / tau1) + (1-a) * np.exp(-1 * t / tau2)
    return y

def fun_scale(t, scale):
    return t * scale

def template_Gen(df, index, bef_aft=0, low_end=45, high_end=300):
    """
    Fit exponential decay parameter from response trace
    :param df: dataframe to obtain response trace
    :param index: row of the dataframe
    :param bef_aft: column of before or after, if 0: Before, if 1: After
    :param low_end: low end of exponential decay trace (peak)
    :param high_end: high end of exponential decay trace

    :return: popt: decay time constant
    """
    # Obtain response
    if bef_aft == 0:
        template = np.mean(np.array(df.Before.loc[index]['trace_y1']), axis=0)[low_end:high_end]
    else:
        template = np.mean(np.array(df.After.loc[index]['trace_y1']), axis=0)[low_end:high_end]

    template = np.squeeze((template - np.min(template)) / np.max(template - np.min(template)))
    template = template[np.argmax(template):]

    xdata = np.arange(len(template))
    popt, _ = curve_fit(fun_exp, xdata, template)

    return popt, xdata, template


def spon_detect(trace, template, plot_low=None, plot_high=None, iffigure=False):
    """
    Detect locations of psc using template matching algorithm
    :param trace: Trace to look for trace
    :param template: Template to perform matching
    :param plot_low:  low bound for plotting
    :param plot_high: high bound for plotting
    :param iffigure: whether to output figure
    :return: list of locations of psc
    """
    trace = -1 * trace

    if plot_low is None:
        plot_low = 0

    if plot_high is None:
        plot_high = len(trace)

    fitted_scale = np.zeros(len(trace) - len(template) + 1)

    for i in range(len(fitted_scale)):
        test_tmp = np.squeeze(trace[i:i + len(template)])
        popt2, _ = curve_fit(fun_scale, template, test_tmp)
        fitted_scale[i] = popt2[0]

    fitted_trace = np.convolve(fitted_scale, template, mode='full') / np.sum(template)
    fitted_trace[np.where(fitted_trace <= 0)[0]] = 0

    standard_err = np.sqrt(np.mean(np.square(fitted_trace - trace)))

    detection_cri = np.divide(fitted_trace, standard_err)

    test_diff = np.diff(detection_cri)
    loc_diff1 = np.where(test_diff > 0.05)[0]
    loc_diff1_plot = loc_diff1[(loc_diff1 >= plot_low) & (loc_diff1 <= plot_high)]
    loc_diff1_plot = np.append(loc_diff1_plot, len(loc_diff1_plot))
    loc_diff_plot = np.diff(loc_diff1_plot)

    loc_plot = np.where(loc_diff_plot > 1)[0]
    loc_plot = loc_plot[detection_cri[loc_diff1_plot[loc_plot]] >= 4]

    # Different template may output different location for the same psc, search around the area
    output_loc = []
    for i in loc_plot:
        output_loc.append(findMax(trace, loc_diff1_plot[i]))
    if iffigure:
        plt.figure()
        plt.plot(detection_cri[plot_low: plot_high])
        plt.plot(trace[plot_low: plot_high] * 1e11, alpha=0.3)
        plt.plot(loc_diff1_plot[loc_plot], np.ones(len(loc_plot)), '*')

    return output_loc


def exp_fit(f, x, y):
    """
    Fit a normalized decay function to exponential decay function f
    :param f: function object
    :param x: x
    :param y: y
    :return:
    popt: fitted parameter, None if no successful fitting
    fit_curve: fitted curve, start from peak
    error: error between raw and fitted trace
    """
    norm_y = (y - np.min(y)) / np.max(y - np.min(y))

    try:
        popt, popcv = curve_fit(f, x, norm_y)
    except:
        return None, None, np.inf

    fit_curve = f(x, *popt) * max(y)
    error = np.nanmean(np.square(fit_curve - y))

    return popt, fit_curve, error


def findMax(trace, loc, width=50):

    start = int(max(loc - width, 0))
    end = int(min(loc + width, len(trace)))
    return start + np.argmax(trace[start:end])


def decay_fit_single(func, trace, reso=25 * 10 ** -6):
    """
    Fit long and short traces to a certain function func
    :param func: function object
    :param trace: long trace
    :param reso: sampling resolution
    :return:
    popt: fitted parameter, None if no successful fitting
    fit_curve: fitted curve, start from peak
    error: error between raw and fitted trace
    """
    xdata = np.arange(len(trace)) * reso
    popt1, fit_curve1, error1 = exp_fit(func, xdata, trace)
    if len(xdata) >= 300:
        popt2, fit_curve2, error2 = exp_fit(func, xdata[:-100], trace[:-100])
        if error2 == min([error1, error2]):
            return popt2, fit_curve2, error2

    return popt1, fit_curve1, error1


def decay_model_select(trace, reso=25 * 10 ** -6):
    """
    fit function to single and double exponential ,select the better one
    :param trace: trace to fit
    :param reso: sampling resolution
    :return:
    popt: fitted parameter, None if no successful fitting
    fit_curve: fitted curve, start from peak
    error: error between raw and fitted trace
    """
    # Fit with single exponential decay
    popt_s, fit_curve_s, error_s = decay_fit_single(fun_exp, trace, reso=reso)
    popt_d, fit_curve_d, error_d = decay_fit_single(fun_2exp, trace, reso=reso)

    if error_s < error_d:
        if popt_s[0] < 0.0003 or error_s > 0.05 * 10 ** -10:
            popt_s[0] = None
        return popt_s[0], fit_curve_s, error_s
    else:
        decay_d = min(np.abs(popt_d[1:]))
        if decay_d < 0.0003 or error_d > 0.05 * 10 ** -10:
            decay_d = None
        return decay_d, fit_curve_d, error_d


def onset_fit(trace, reso=25 * 10 ** -6):
    """
    Obtain onset time constant as the time difference from 20% to 80% of peak amplitude
    :param trace: trace to fit
    :param reso: sampling resolution
    :return: onset time constant
    """
    onset_trace = trace[:np.argmax(trace)]
    high_thres = onset_trace[onset_trace < np.max(trace) * 0.8]
    low_thres = onset_trace[onset_trace < np.max(trace) * 0.2]
    if len(high_thres) == 0 or len(low_thres) == 0:
        return None
    high_end = np.where(onset_trace == high_thres[-1])[0]
    low_end = np.where(onset_trace == low_thres[0])[0]

    return (high_end - low_end) * reso


def psc_search(trace, loc_ori):
    """
    Obtain psc parameter for all detected pscs in location list
    :param trace: raw trace
    :param loc_ori: detected psc location list
    :return: psc object containing raw trace, amplitude, onset time constant, decay time constant, decay fit and error
    """
    loc = [int(i) for i in loc_ori[0]]
    loc = np.append(loc,
                    np.iinfo(np.int32).max)  # If two traces are very close, use the next loc as the end of previous
    trace = -1 * trace
    psc_seq = []
    for i in range(len(loc) - 1):
        psc = {}
        peak_loc = np.argmax(trace[max(int(loc[i]) - 50, 0):min(int(loc[i]) + 50, len(trace))])
        psc['trace'] = trace[max(loc[i] - 50 + peak_loc - 50, 0):min(
            [loc[i] - 50 + peak_loc + 300, loc[i + 1] - 50, len(trace)])]
        if np.argmax(psc['trace']) > 200:
            continue
        if len(psc['trace']) < 200:  # Control the distance of two
            continue
        if np.nanmean(psc['trace'][:10]) > 0.5 * psc['amp']:  # Control noisy level
            continue
        psc['amp'] = max(psc['trace']) - np.mean(psc['trace'][:10])
        psc['onset_tau'] = onset_fit(psc['trace'])
        decay_tau, fitted_curve, error = decay_model_select(psc['trace'][peak_loc:])
        psc['decay_tau'] = decay_tau
        psc['decay_fit'] = fitted_curve
        psc['fit_error'] = error
        psc_seq.append(psc)

    return psc_seq