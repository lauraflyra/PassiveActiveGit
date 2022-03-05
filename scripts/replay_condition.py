"""
Code from Laura Lyra, 02.03.2022
"""

import numpy as np
import math
import pandas as pd
from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit


# both savgol and upsampling need to be applied for every image and every participant
# 1. Smooth the data with a Savitzky-Golay filter with polynomial order 2 and window size of 13ms.

def sav_gol(df, **kwargs):
    """
    Smooth the data with a Savitzky-Golay filter
    :param df: data frame with columns x and y to be filtered
    :param \**kwargs: arguments for the signal.savgol_filter
    :return: modified copy df with columns x and y filtered
    """
    savgol_x = signal.savgol_filter(df.x, **kwargs)
    savgol_y = signal.savgol_filter(df.y, **kwargs)

    pd.options.mode.chained_assignment = None  # default='warn'
    df_copy = df.copy()
    df_copy.loc[:, "x"] = savgol_x
    df_copy.loc[:, "y"] = savgol_y
    return df_copy


# 2. Upsample the data (by linear interpolation) to match our projector frequency of 1440Hz
# Upsampling is now done with scipy.interpolate.interp1d,
# that behaves better than pandas.resample.interpolate

def upsampling(df):
    """

    """
    df_time = df.set_index(pd.to_timedelta(df.index, unit='ms'))  # put the index as time in ms

    # Resample two data frames, one interpolating the x and y with scipy.interpolate and the other doing ffil() on pandas.resample,
    # i.e, forward fill method that uses the last known value to replace the NaN.

    time_original = df_time.index.to_numpy(dtype=float) * 1e-9  # now we have an array in seconds

    x_original = df_time.x.to_numpy(dtype=float)
    y_original = df_time.y.to_numpy(dtype=float)

    df_final = df_time.drop(columns=["x", "y"])
    df_final = df_final.resample('0.694444L').ffill()

    time_resampled = df_final.index.to_numpy(dtype=float) * 1e-9

    f_interpol_x = interpolate.interp1d(time_original, x_original)
    f_interpol_y = interpolate.interp1d(time_original, y_original)

    x_resampled = f_interpol_x(time_resampled)
    y_resampled = f_interpol_y(time_resampled)

    df_final.insert(2, "x", x_resampled)
    df_final.insert(3, "y", y_resampled)

    return df_final


# after upsampling, the dt between two rows is changed.
dt = 0.694444  # ms


# velocity is gonna be in dva/ms

# contrast needs to be handled in two steps: First insert the contrast for the invalid saccades and
# fixations, then insert the contrast for the remaining saccades.

# In order to add the contrast ramp for the valid saccades, we fit a Gumbel CDF to the position deltas
# some auxiliary functions are needed:

# Transform position to change in position, so we can fit a curve to change in position and
# then take the derivative for the velocity profile.
def delta_position(df):
    """
    :param df: data frame with columns x and y to be modified to change in position,
    here we expect a sliced data frame containing only one time sequence for one saccade.
    """
    x = df.x
    y = df.y
    x_on = x[0]
    y_on = y[0]
    delta_pos = np.sqrt((x - x_on) ** 2 + (y - y_on) ** 2)
    return delta_pos


def func_Gumbel(t, t0, b):
    """
    Fit Gumbel CDF, amplitude is out, because we assume data is between 0 and 1.
    """
    return  np.exp(-np.exp(-(t - t0) / b))


# we want contrast valued to always be between 0 and 1
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def norm_inverse_gauss(x, mean=100, std=35):
    """
    We need to have a Gaussian fitted for the invalid contrasts. Mean is typically 100ms, and
    35ms as std looked visually adequate.
    :return: Gaussian with values between 0 and 1
    """
    return NormalizeData(-(1 / (np.sqrt(2 * np.pi * std ** 2))) * (np.exp(-(x - mean) ** 2 / (2 * std ** 2))))


def add_contrast(df_upsampled):
    """
    Add contrast column, with ramping contrast during saccades and invalid fixations. For valid
    saccades, the contrast ramp is calculated in steps:
    1. Put position into position deltas
    2. Fit a Gumbel CDF to it
    3. Take the derivative of the fitted data.
    4. Invert and normalize the profile encoutered.
    For invalid saccades and fixations, a Gaussian is fitted. The contrast ramp in this case is limited
    to 100ms down or up. For invalid intervals that last more than that, we keep the contrast at zero
    in the middle.
    :param df_upsampled: data frame containing colums x,y, invalid and TimeDeltaIndex. Already upsampled
    to 1440Hz.
    :return: data frame with one more column filled with appropriate contrast values.
    """
    # contrast needs to be handled in two steps: First insert the contrast for the invalid saccades and
    # fixations, then insert the contrast for the remaining saccades.

    contrast = np.ones(len(df_upsampled))
    df_upsampled.insert(7, "contrast", contrast)

    ident = np.unique(np.asarray(df_upsampled.identifier))

    # In order to fit the Gaussian to the invalid sections we need:
    mean = 100
    std = 35
    # get the number of time steps that correspond to 100ms
    n_t_steps = int(mean / dt)

    for ids in ident:
        df_temp = df_upsampled[(df_upsampled.identifier == ids) & (df_upsampled.invalid == 1)]
        if len(df_temp) > 0:
            start = df_temp.index[0]
            end = df_temp.index[-1]
            dur_invalid = (end - start).total_seconds() * 1e3  # duration of invalid interval in ms
            # we need to check if this is bigger than 200ms
            tot_t_steps = len(df_temp)  # get the total number of invalid time steps

            if dur_invalid / 2 > mean:
                x_gauss_down = np.linspace(0, mean, n_t_steps)
                contrast_gauss = norm_inverse_gauss(x_gauss_down, mean, std)
                plateau = np.zeros(tot_t_steps - 2 * n_t_steps)
                x_gauss_up = np.linspace(mean, 2 * mean, n_t_steps)
                contrast_gauss_up = norm_inverse_gauss(x_gauss_up, mean, std)
                contrast_temp = np.append(contrast_gauss, plateau)
                contrast_invalid = np.append(contrast_temp, contrast_gauss_up)
                assert (len(contrast_invalid) == tot_t_steps)
            else:
                x_gauss = np.linspace(0, 200, tot_t_steps)
                contrast_invalid = norm_inverse_gauss(x_gauss, mean, std)

            df_upsampled.loc[(df_upsampled.identifier == ids) & (df_upsampled.invalid == 1),
                             "contrast"] = contrast_invalid

    # Now we loop over ids again, but this time for the valid saccades


###################################
    # maybe I need to separate this into two functions!
    # also need to find a way to handle when the parameters are not found.
    for ids in ident:
        df_valid = df_upsampled[(df_upsampled.identifier == ids) &
                                (df_upsampled.is_saccade == 1) & (df_upsampled.invalid == 0)]
        print(ids)
        if len(df_valid) > 0:
            print("in")
            tot_val_steps = len(df_valid)
            delta_df = delta_position(df_valid)
            delta_df = delta_df / np.max(delta_df) # make the delta df be between 0 and 1
            t_delta = np.asarray(delta_df.index.total_seconds()) * 1e3  # time in ms
            t_delta = t_delta - t_delta[0]
            # fit Gumbel CDF
            popt_Gumbel, _ = curve_fit(func_Gumbel, t_delta, delta_df)
            # Normalize the derivative after inverting
            contrast_valid = NormalizeData(-np.diff(func_Gumbel(t_delta, *popt_Gumbel)))
            contrast_valid = np.concatenate((contrast_valid, [1]))

            assert (len(contrast_valid) == tot_val_steps)

            df_upsampled.loc[(df_upsampled.identifier == ids) &
                             (df_upsampled.is_saccade == 1) & (df_upsampled.invalid == 0),
                             "contrast"] = contrast_valid
        return df_upsampled
