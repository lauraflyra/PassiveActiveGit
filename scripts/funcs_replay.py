import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

# both savgol and upsampling need to be applied for every image and every participant
# 1st  Smooth the data with a Savitzky-Golay filter with polynomial order 2 and window size of 13ms.

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

# 2nd 2. Upsample the data (by linear interpolation) to match our projector frequency of 1440Hz
# Upsampling is now done with scipy.interpolate.interp1d,
# that behaves better than pandas.resample.interpolate

def upsampling(df):
    """
    Upsample the data (by linear interpolation) to match our projector frequency of 1440Hz
    :param df: data frame with sampling in a frequency of 1000Hz
    :return: data frame with resampled columns. Columns x and y had linear interpolation and the others
    are completed with forward fill method.
    """

    # I can probably place this in the iteration of the replay condition

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


