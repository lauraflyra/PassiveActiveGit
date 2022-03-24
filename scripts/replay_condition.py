"""
Code from Laura Lyra, 02.03.2022
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import norm
from itertools import groupby
from operator import itemgetter
import warnings



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

    f_interpol_x = interpolate.interp1d(time_original, x_original, fill_value="extrapolate")
    f_interpol_y = interpolate.interp1d(time_original, y_original, fill_value="extrapolate")

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
    return np.exp(-np.exp(-(t - t0) / b))


# we want contrast valued to always be between 0 and 1
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def NormalizeInterval(data,a,b):
    """
    Normalize data to interval [a,b]
    """
    return (b-a)*(data - np.min(data)) / (np.max(data) - np.min(data)) + a


def norm_inverse_gauss(x, mean=100, std=35):
    """
    We need to have a Gaussian fitted for the invalid contrasts. Mean is typically 100ms, and
    35ms as std looked visually adequate.
    :return: Gaussian with values between 0 and 1
    """
    return NormalizeData(-(1 / (np.sqrt(2 * np.pi * std ** 2))) * (np.exp(-(x - mean) ** 2 / (2 * std ** 2))))

def find_nearest(array, value):
    """Find index of nearest value in numpy array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def add_contrast_valid_apparent_case(df):
    """
    HERE: APPARENT CASE

    We assume continuous contrast was already inserted in the data frame. Meaning that we copy the contrast values for
    the invalid saccades to the new column contrast_app, refering to the apparent contrast condition.

    Changes in x,y coordinates, such that in the first half of the contrast ramp we have x,y
    coordinates fixed to what they were before the beggining of the saccade. In the seccond half, the x,y
    coordinates are fixed to the fixation after the saccade ends.

    Add contrast column, with ramping contrast during valid saccades. Here, the contrast ramp is calculated in steps:
    1. Put position into position deltas
    2. Fit a Gumbel CDF to it
    3. Take the derivative of the fitted data.
    4. Invert and normalize the profile encoutered.
    :param df: data frame containing colums x,y, invalid and TimeDeltaIndex. Already upsampled
    to 1440Hz and already containing contrast values for invalid saccades in a contrast_cont column.
    :return: data frame with one more column filled with appropriate contrast values for the valid saccades. x_apparent
    and y_apparent are also included.
    """

    df_upsampled = df.copy()

    try:
        contrast = df_upsampled.contrast_cont
    except:
        contrast = np.ones(len(df_upsampled))

    original_x = df_upsampled.x
    original_y = df_upsampled.y

    try:
        df_upsampled.insert(7, "contrast_app", contrast)
    except:
        pass

    try:
        df_upsampled.insert(4, "x_apparent", original_x)
        df_upsampled.insert(5, "y_apparent", original_y)
    except:
        print("x,y didnt work")

    ident = np.unique(np.asarray(df_upsampled.identifier))

    warnings.filterwarnings('ignore')

    params = []
    sacc_durations = []
    for count, ids in enumerate(ident):
        df_valid = df_upsampled[(df_upsampled.identifier == ids) &
                                (df_upsampled.is_saccade == 1) & (df_upsampled.invalid == 0)]
        if len(df_valid) > 0:
            tot_val_steps = len(df_valid)
            delta_df = delta_position(df_valid)
            delta_df = delta_df / np.max(delta_df)
            t_delta = np.asarray(delta_df.index.total_seconds()) * 1e3  # time in ms
            t_delta = t_delta - t_delta[0]

            # we need to chose initial guesses for our t0 and b
            # for that we do a gridsearch, getting the Gumbel to every possible combination
            # and then doing the residuals squared sum
            # Empirically it looked ok to:
            # 1. Vary t0 initial value from 1/4 to 3/4 of the saccade duration.
            # 2. Vary b from 1/saccade_duration to saccade_duration/2
            t0_range = np.linspace(t_delta[-1] / 4, 3 * t_delta[-1] / 4)
            b_range = np.linspace(1 / t_delta[-1], t_delta[-1] / 2)
            tt, bb = np.meshgrid(t0_range, b_range, sparse=True)

            gumbel_meshed = []
            for t in t_delta:
                zz = func_Gumbel(t, tt, bb)
                gumbel_meshed.append(zz)

            gumbel_meshed = np.asarray(gumbel_meshed)
            residuals = np.sum((gumbel_meshed - delta_df[:, np.newaxis, np.newaxis]) ** 2, axis=0)

            # we take as initial conditions the combination that has minimum residual
            idx_min = np.unravel_index(np.argmin(residuals, axis=None), residuals.shape)
            t0_initial = tt[0, idx_min[0]]
            b_initial = bb[idx_min[1], 0]

            #### HERE CHECK IF A SOLUTION WAS FOUND, IF NOT DO SOMETHING ELSE
            from scipy.optimize import OptimizeWarning
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    # fit Gumbel CDF
                    popt_Gumbel, _ = curve_fit(func_Gumbel, t_delta, delta_df, p0=[t0_initial, b_initial])
                    if np.any(popt_Gumbel>1e2) or np.any(popt_Gumbel < 1e-2):
                        raise OptimizeWarning

                    params.append(popt_Gumbel)
                    sacc_durations.append(t_delta[-1])
                    # Normalize the derivative after inverting
                    contrast_valid = NormalizeInterval(-np.diff(func_Gumbel(t_delta, *popt_Gumbel)), a=0, b=0.999998)
                    contrast_valid = np.concatenate((contrast_valid, [0.999999]))

                except (OptimizeWarning,RuntimeError):
                    last_sacc_duration = t_delta[-1]
                    try:
                        idx_nearest = find_nearest(sacc_durations, last_sacc_duration)
                        approx_gumbel_params = params[idx_nearest]
                        contrast_valid = NormalizeInterval(-np.diff(func_Gumbel(t_delta, *approx_gumbel_params)), a=0,
                                                           b=0.999998)
                        contrast_valid = np.concatenate((contrast_valid, [0.999999]))
                        if (np.isnan(contrast_valid).any()):
                            raise ValueError

                    except ValueError:

                        # If there's nothing to find, we just go with the initial guesses.
                        contrast_valid = NormalizeInterval(-np.diff(func_Gumbel(t_delta, t0_initial, b_initial)), a=0,
                                                           b=0.999998)
                        contrast_valid = np.concatenate((contrast_valid, [0.999999]))

            assert (len(contrast_valid) == tot_val_steps)

            df_upsampled.loc[(df_upsampled.identifier == ids) &
                             (df_upsampled.is_saccade == 1) & (df_upsampled.invalid == 0),
                             "contrast_app"] = contrast_valid

            #### Now that we inserted the contrast values, we can adjust the x,y positions

            first_saccade_timestamp = df_valid.time[0]
            last_saccade_timestamp = df_valid.time[-1]

            # We take the [0] component because the upsampling sometimes leaves us with repeated
            # timestamps, which means we take the first value the corresponds to that timestamp.

            start_x = df_upsampled.loc[df_upsampled.time == first_saccade_timestamp - 1].x[0]
            start_y = df_upsampled.loc[df_upsampled.time == first_saccade_timestamp - 1].y[0]

            try:
                end_x = df_upsampled.loc[df_upsampled.time == last_saccade_timestamp + 1].x[0]
                end_y = df_upsampled.loc[df_upsampled.time == last_saccade_timestamp + 1].y[0]
            except IndexError:
                end_x = df_upsampled.loc[df_upsampled.time == last_saccade_timestamp].x[0]
                end_y = df_upsampled.loc[df_upsampled.time == last_saccade_timestamp].y[0]


            timestamp_zero_contrast = int(df_upsampled[(df_upsampled.is_saccade == 1) &
                                                       (df_upsampled.identifier == ids) & (df_upsampled.invalid == 0) &
                                                       (df_upsampled.contrast_app == 0)].time)

            df_upsampled.loc[(df_upsampled.time >= first_saccade_timestamp) &
                             (df_upsampled.time < timestamp_zero_contrast), 'x_apparent'] = start_x

            df_upsampled.loc[(df_upsampled.time >= first_saccade_timestamp) &
                             (df_upsampled.time < timestamp_zero_contrast), 'y_apparent'] = start_y

            df_upsampled.loc[(df_upsampled.time >= timestamp_zero_contrast) &
                             (df_upsampled.time <= last_saccade_timestamp), 'x_apparent'] = end_x

            df_upsampled.loc[(df_upsampled.time >= timestamp_zero_contrast) &
                             (df_upsampled.time <= last_saccade_timestamp), 'y_apparent'] = end_y

    return df_upsampled


def add_contrast_invalid(df):
    """
    REMEMBER: df is one image for one participant!
    Add contrast column, with ramping contrast during invalid saccades and fixations.
    Adding only contrast to invalid saccades and fixations corresponds to the continuous condition, thus we add
    the values in a contrast_cont column.

    THIS FUNCTION IS EXPECTED TO BE RUN BEFORE THE add_contrast_valid_apparent_case FUNCTION!

    For invalid saccades and fixations, a Gaussian CDF is fitted. We consider mean 50 and std = 50/3.
    The contrast ramp in this case is limited to 100ms down or up.
    For invalid intervals that last more than that, we keep the contrast at zero
    in the middle.
    :param df: data frame containing colums x,y, invalid and TimeDeltaIndex. Already upsampled
    to 1440Hz.
    :return: data frame with one more column filled with appropriate contrast values for invalid saccades.
    """
    # contrast needs to be handled in two steps: First insert the contrast for the invalid saccades and
    # fixations, then insert the contrast for the remaining saccades.

    df_upsampled = df.copy()
    contrast = np.ones(len(df_upsampled))
    try:
        df_upsampled.insert(7, "contrast_cont", contrast)
    except:
        pass
    # In order to fit the Gaussian cdf to the invalid sections we need:
    # we want to get a ramp that lasts 100ms to go up or down
    max_ramp = 100
    # get the number of time steps that correspond to 100ms
    n_t_steps = int(max_ramp / dt)

    # First: get the indexes of all invalid time stamps
    invalid_stamps = np.where(df_upsampled[df_upsampled.invalid == 1].time.diff().fillna(1) <= 1)[0]
    # Second: get a list where the indexes are grouped in sequences of consecutive numbers
    idx_invalid = []
    for k, g in groupby(enumerate(invalid_stamps), lambda ix: ix[0] - ix[1]):
        idx_invalid.append(list((map(itemgetter(1), g))))

    # Here if the length of idx_invalid is zero, it doesn't even enter the loop
    for i in range(len(idx_invalid)):
        begin_invalid, end_invalid = idx_invalid[i][0], idx_invalid[i][-1]
        duration_invalid = end_invalid - begin_invalid + 1

        index_want = df_upsampled[(df_upsampled.invalid == 1)][begin_invalid:end_invalid + 1].index

        # check if the last index from the interval is also the last time stamp from that trial
        last_index = df_upsampled.loc[index_want[-1]].time
        last_time = df_upsampled.time[-1]
        # if they are equal, I don't ramp up the contrast, I only ramp it down
        if last_index == last_time:
            x_cdf_down = np.linspace(0, max_ramp, np.min([duration_invalid, n_t_steps]))
            contrast_cdf_down = -norm(loc=50, scale=50 / 3).cdf(x_cdf_down) + 1
            if duration_invalid > n_t_steps:
                plateau = np.zeros(duration_invalid - n_t_steps)
                contrast_invalid = np.append(contrast_cdf_down, plateau)
            else:
                contrast_invalid = contrast_cdf_down
        else:
            if duration_invalid / 2 > n_t_steps:
                # If the whole duration of the invalid interval is bigger than 200ms, we want to have a plateau
                # in the middle, where contrast is set to zero.

                # Fit a Gaussian cdf to the contrast there. Gaussian is assumed to have mean = 50ms and
                # std = 50/3
                x_cdf_down = np.linspace(0, max_ramp, n_t_steps)
                # -norm and +1 are used to get contrast ramping from 1 down to 0
                contrast_cdf_down = -norm(loc=50, scale=50 / 3).cdf(x_cdf_down) + 1

                plateau = np.zeros(duration_invalid - 2 * n_t_steps)

                x_cdf_up = np.linspace(max_ramp, 2 * max_ramp, n_t_steps)
                # here a cdf ramps contrast from 0 back to 1, mean is adjusted such that we have a cdf for
                # values between 100 and 200ms.
                contrast_cdf_up = norm(loc=50 + max_ramp, scale=50 / 3).cdf(x_cdf_up)

                contrast_temp = np.append(contrast_cdf_down, plateau)
                contrast_invalid = np.append(contrast_temp, contrast_cdf_up)
                contrast_invalid = NormalizeInterval(contrast_invalid, a=0, b=0.999998)
                assert (len(contrast_invalid) == duration_invalid)
            else:
                # if the whole duration of the invalid interval is smaller than 200ms, we don't need a plateau.

                x_cdf_down = np.linspace(0, max_ramp, int(duration_invalid / 2))
                contrast_cdf_down = -norm(loc=50, scale=50 / 3).cdf(x_cdf_down) + 1

                x_cdf_up = np.linspace(max_ramp, 2 * max_ramp, duration_invalid - len(x_cdf_down))
                contrast_cdf_up = norm(loc=50 + max_ramp, scale=50 / 3).cdf(x_cdf_up)
                contrast_invalid = np.append(contrast_cdf_down, contrast_cdf_up)
                contrast_invalid = NormalizeInterval(contrast_invalid,a=0,b=0.999998)

        df_upsampled.loc[(df_upsampled.invalid == 1) &
                         (df_upsampled.index.isin(index_want)), "contrast_cont"] = contrast_invalid

    return df_upsampled


def add_contrast_valid(df):
    """
    HERE: CONTINUOS + CONTRAST CASE
    NOT USED IN THE CURRENT DATA FRAMES. Can used with slight adaptations if we want to have this case.
    No changes in x,y coordinates.
    Add contrast column, with ramping contrast during valid saccades. Here, the contrast ramp is calculated in steps:
    1. Put position into position deltas
    2. Fit a Gumbel CDF to it
    3. Take the derivative of the fitted data.
    4. Invert and normalize the profile encoutered.
    :param df: data frame containing colums x,y, invalid and TimeDeltaIndex. Already upsampled
    to 1440Hz.
    :return: data frame with one more column filled with appropriate contrast values for the valid saccades.
    """
    # contrast needs to be handled in two steps: First insert the contrast for the invalid saccades and
    # fixations, then insert the contrast for the remaining saccades.
    df_upsampled = df.copy()
    contrast = np.ones(len(df_upsampled))

    try:
        df_upsampled.insert(7, "contrast", contrast)
    except:
        pass

    ident = np.unique(np.asarray(df_upsampled.identifier))

    warnings.filterwarnings('ignore')

    params = []
    sacc_durations = []
    for count, ids in enumerate(ident):
        df_valid = df_upsampled[(df_upsampled.identifier == ids) &
                                (df_upsampled.is_saccade == 1) & (df_upsampled.invalid == 0)]
        if len(df_valid) > 0:
            tot_val_steps = len(df_valid)
            delta_df = delta_position(df_valid)
            delta_df = delta_df / np.max(delta_df)
            t_delta = np.asarray(delta_df.index.total_seconds()) * 1e3  # time in ms
            t_delta = t_delta - t_delta[0]
            sacc_durations.append(t_delta[-1])

            # we need to chose initial guesses for our t0 and b
            # for that we do a gridsearch, getting the Gumbel to every possible combination
            # and then doing the residuals squared sum
            # Empirically it looked ok to:
            # 1. Vary t0 initial value from 1/4 to 3/4 of the saccade duration.
            # 2. Vary b from 1/saccade_duration to saccade_duration/2
            t0_range = np.linspace(t_delta[-1] / 4, 3 * t_delta[-1] / 4)
            b_range = np.linspace(1 / t_delta[-1], t_delta[-1] / 2)
            tt, bb = np.meshgrid(t0_range, b_range, sparse=True)

            gumbel_meshed = []
            for t in t_delta:
                zz = func_Gumbel(t, tt, bb)
                gumbel_meshed.append(zz)

            gumbel_meshed = np.asarray(gumbel_meshed)
            residuals = np.sum((gumbel_meshed - delta_df[:, np.newaxis, np.newaxis]) ** 2, axis=0)

            # we take as initial conditions the combination that has minimum residual
            idx_min = np.unravel_index(np.argmin(residuals, axis=None), residuals.shape)
            t0_initial = tt[0, idx_min[0]]
            b_initial = bb[idx_min[1], 0]

            # fit Gumbel CDF
            popt_Gumbel, _ = curve_fit(func_Gumbel, t_delta, delta_df, p0=[t0_initial, b_initial])
            params.append(popt_Gumbel)
            # Normalize the derivative after inverting
            contrast_valid = NormalizeData(-np.diff(func_Gumbel(t_delta, *popt_Gumbel)))
            contrast_valid = np.concatenate((contrast_valid, [1]))

            assert (len(contrast_valid) == tot_val_steps)

            df_upsampled.loc[(df_upsampled.identifier == ids) &
                             (df_upsampled.is_saccade == 1) & (df_upsampled.invalid == 0),
                             "contrast"] = contrast_valid

    return df_upsampled
