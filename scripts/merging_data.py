"""
Code from Laura Lyra, 02.11.2021
"""

import numpy as np
import math
import pandas as pd


def get_valid_times(raw_data, corpus_data):
    """
    get all valid time stamps, i.e, filter the times such
    that all times in the raw data are in between the imagestart and imageend times
    in the corpus_data.
    Here corpus_data corresponds to the data already filtered by participant number to match
    the participant number from the raw_data file.
    :param raw_data: pandas data frame with columns = =['time', 'x', 'y', 'pupil']
    :param corpus_data: pandas data frame filtered for participant number #n

    :return: raw_data with only valid time stamps
    """


    trials = np.unique(corpus_data['trialno'])
    n_trials = len(trials)
    intervals = np.zeros((n_trials, 2))
    for idx, trial in enumerate(trials):
        intervals[idx, 0] = corpus_data[corpus_data['trialno'] == trial]["imagestart"].iloc[0]
        intervals[idx, 1] = corpus_data[corpus_data['trialno'] == trial]["imageend"].iloc[0]

    ##subtract time zero
    time_zero = corpus_data["imagestart"].iloc[0]
    raw_data['time'] -= time_zero
    raw_data = raw_data.drop(raw_data[raw_data.time < 0].index)
    raw_data['time'] += time_zero

    for trial in range(1, n_trials):
        end_last = intervals[trial - 1, 1]  # end of the last trial
        beg_curr = intervals[trial, 0]  # beggining of the current trial
        raw_data = raw_data.drop(raw_data[(raw_data.time > end_last) & (raw_data.time < beg_curr)].index)

    raw_data = raw_data.drop(raw_data[(raw_data.time > intervals[-1, 1])].index)

    return raw_data, np.asarray(intervals, dtype=int)

def linear_transf_pix_dva(raw_data, pix_to_dva=0.037):
    """
    Get the raw data and transform the x-y coordinates from pixel space to dva space
    using a linear transformation.
    :param raw_data: raw data with x-y coordinates in pixels
    :param pix_to_dva: value of 1 pixel in degrees visual angle

    :return: data frame with the same shape and columns as raw_data, but with x-y
    coordinates in dva
    """
    data_dva = raw_data.copy()
    data_dva.x = data_dva.x * pix_to_dva
    data_dva.y = data_dva.y * pix_to_dva
    return data_dva


def associate_trial_info(filtered_raw, intervals, corpus_data):
    """
    Associate each time stamp and trial to the correspoding image number, filter type and region,
    presence of target and expected location, according to the corpus data.

    :param filtered_raw: pandas data frame with columns = =['time', 'x', 'y', 'pupil'] after
    filtering for the valid times with the function get_valid_times
    :param intervals: numpy array n_trials x 2, intervals of valid times for each trial
    :param corpus data: pandas data frame filtered for participant number #n

    :return: filtered_raw with 5 more columns: 'imageno', 'filtertype','filterregion',
    'targetpresent' and 'expectedlocation', trial durations in numpy array length
    n_trials, each entry is the number of time stamps that correspond to one trial
    """
    # associate image number (imageno) to the appropriate trial condition
    n_trials = intervals.shape[0]
    trials = np.unique(corpus_data['trialno'])
    trial_durations = np.zeros(n_trials, dtype=int)
    raw_imagenos = []
    filter_type = []
    filter_region = []
    target_pres = []
    expected_loc = []
    for idx, trial in enumerate(trials):
        im = corpus_data[corpus_data['trialno'] == trial]["imageno"].iloc[0]
        filtertype = corpus_data[corpus_data['trialno'] == trial]["filtertype"].iloc[0]
        filterreg = corpus_data[corpus_data['trialno'] == trial]["filterregion"].iloc[0]
        targetpres = corpus_data[corpus_data['trialno'] == trial]["targetpresent"].iloc[0]
        expectedloc = corpus_data[corpus_data['trialno'] == trial]["expectedlocation"].iloc[0]

        trial_durations[idx] = len(filtered_raw[(filtered_raw.time >= intervals[idx][0]) &
                                                (filtered_raw.time <= intervals[idx][1])])

        temp = im * np.ones([trial_durations[idx]])
        temp_ft = filtertype * np.ones([trial_durations[idx]])
        temp_fr = filterreg * np.ones([trial_durations[idx]])
        temp_tp = targetpres * np.ones([trial_durations[idx]])
        temp_el = expectedloc * np.ones([trial_durations[idx]])

        raw_imagenos.extend(temp)
        filter_type.extend(temp_ft)
        filter_region.extend(temp_fr)
        target_pres.extend(temp_tp)
        expected_loc.extend(temp_el)

    try:
        filtered_raw.insert(4, "imageno", np.asarray(raw_imagenos, dtype=int))
        filtered_raw.insert(5, "filtertype", np.asarray(filter_type, dtype=int))
        filtered_raw.insert(6, "filterregion", np.asarray(filter_region, dtype=int))
        filtered_raw.insert(7, "targetpresent", np.asarray(target_pres, dtype=int))
        filtered_raw.insert(8, "expectedlocation", np.asarray(expected_loc, dtype=int))
    except ValueError:
        print("Columns already inserted")
        filtered_raw.imageno = np.asarray(raw_imagenos, dtype=int)
        filtered_raw.filtertype = np.asarray(filter_type, dtype=int)
        filtered_raw.filterregion = np.asarray(filter_region, dtype=int)
        filtered_raw.expectedlocation = np.asarray(expected_loc, dtype=int)

    return filtered_raw, trial_durations


def add_sacc_val_id(filtered_raw, corpus_data):
    """
    By comparing time stamps with the corpus_data, we get which time stamps corresponds
    to saccades (=1) and which dont (=0). Add an id that corresponds to 3 digits of subjet number
    + 3 digits trial number + 2 digits fixation/saccade number. Also add information about invalid
    saccades/fixations, in invalid column where valid=0 and invalid=1.

    :param filtered_raw: pandas data frame with columns = =['time', 'x', 'y', 'pupil','imageno']
    :param corpus data: pandas data frame filtered for participant number #n

    :return: filtered_raw with three more columns: 'is_saccade', that is composed by zeros and ones,
    where zero correspond to fixation and a one correspond to a saccade, 'identifier' and 'invalid'.

    """

    pd.options.mode.chained_assignment = None  # default='warn'

    inds = pd.isnull(corpus_data.sacno)
    corpus_data.loc[inds, "sacinvalid"] = 0

    invalid_sacfix = ((corpus_data.fixinvalid + corpus_data.sacinvalid) >= 1).astype(int)

    identifier = np.zeros(len(filtered_raw.time), dtype=str)
    is_saccade = np.ones(len(filtered_raw.time), dtype=int)
    invalid = np.zeros(len(filtered_raw.time), dtype=int)

    try:
        filtered_raw.insert(0, "identifier", identifier)
        filtered_raw.insert(7, "is_saccade", is_saccade)
        filtered_raw.insert(7, "invalid", invalid)
    except:
        pass

    corpus_data_noNaN = corpus_data[corpus_data.sacno.notnull()]

    subject = np.unique(corpus_data.subject)[0]
    for image in np.unique(filtered_raw.imageno):
        imagestart = filtered_raw[filtered_raw.imageno == image].time.iloc[0]

        # First we just try to get the invalid saccades
        corpus_invalid = corpus_data[invalid_sacfix & (corpus_data.imageno == image)]
        invalid_sac_no = np.unique(corpus_invalid.sacno[corpus_invalid.sacno.notnull()])

        for insac in invalid_sac_no:
            invalidity = corpus_invalid[corpus_invalid.sacno == insac].sacinvalid
            in_saconset = corpus_invalid[corpus_invalid.sacno == insac].saconset
            in_sacoffset = corpus_invalid[corpus_invalid.sacno == insac].sacoffset

            filtered_raw.loc[(filtered_raw.time >= imagestart + int(in_saconset)) &
                             (filtered_raw.time <= imagestart + int(in_sacoffset)), "invalid"] = int(invalidity)

        # Second we get invalid fixations + introduce the ids
        trialno = corpus_data[corpus_data.imageno == image].trialno.iloc[0]
        fix_nbs = corpus_data[(corpus_data.imageno == image)].fixno
        for count, fix in enumerate(fix_nbs):

            if math.isnan(fix):
                # this is to make sure that fixations that have NaNs have the id
                # corresponding to the saccade in that same row and are invalid/valid
                # for the duration of that saccade
                c_temp = corpus_data[corpus_data.imageno == image].copy()
                sacno = c_temp[np.logical_not(c_temp.fixno.notnull())].sacno.iloc[0]
                saconset = c_temp[np.logical_not(c_temp.fixno.notnull())].saconset.iloc[0]
                sacoffset = c_temp[np.logical_not(c_temp.fixno.notnull())].sacoffset.iloc[0]
                if saconset == 1:
                    saconset = 0
                invalidity = c_temp[np.logical_not(c_temp.fixno.notnull())].sacinvalid.iloc[0]
                ident = "" + str(subject).zfill(3) + str(trialno).zfill(3) + str(int(sacno)).zfill(2) + ""
                filtered_raw.loc[(filtered_raw.time >= imagestart + saconset) &
                                 (filtered_raw.time <= imagestart + sacoffset), "identifier"] = ident
                filtered_raw.loc[(filtered_raw.time >= imagestart + saconset) &
                                 (filtered_raw.time <= imagestart + sacoffset), "invalid"] = int(invalidity)


            else:
                # This else handles assigning saccades and invalidity
                invalidity = int(corpus_data[(corpus_data.imageno == image) & (corpus_p1.fixno == fix)].fixinvalid)
                ident = "" + str(subject).zfill(3) + str(trialno).zfill(3) + str(int(fix)).zfill(2) + ""
                fixonset = int(corpus_data[(corpus_data.imageno == image) & (corpus_data.fixno == fix)].fixonset)
                # necessary if because in the corpus data their time starts at 1.
                if fixonset == 1:
                    fixonset = 0
                fixoffset = int(corpus_data[(corpus_data.imageno == image) & (corpus_data.fixno == fix)].fixoffset)
                filtered_raw.loc[(filtered_raw.time >= imagestart + fixonset) &
                                 (filtered_raw.time <= imagestart + fixoffset), "is_saccade"] = 0
                filtered_raw.loc[(filtered_raw.time >= imagestart + fixonset) &
                                 (filtered_raw.time <= imagestart + fixoffset), "invalid"] = invalidity

                # This handles assigning ids
                if count < len(corpus_data[(corpus_data.imageno == image)].fixno) - 1:
                    next_fix = int(corpus_data[(corpus_data.imageno == image)].fixno.iloc[count + 1])
                    next_fixonset = int(
                        corpus_data[(corpus_data.imageno == image) & (corpus_data.fixno == next_fix)].fixonset)
                    filtered_raw.loc[(filtered_raw.time >= imagestart + fixonset) &
                                     (filtered_raw.time < imagestart + next_fixonset), "identifier"] = ident


                else:
                    filtered_raw.loc[(filtered_raw.time >= imagestart + fixonset) &
                                     (filtered_raw.time <= imagestart + fixoffset), "identifier"] = ident

        try:
            assert np.sum(corpus_data_noNaN[corpus_data_noNaN.imageno == image].sacdur) == np.sum(
                filtered_raw[filtered_raw.imageno == image].is_saccade != 0)
            assert np.sum(corpus_data[corpus_data.imageno == image].fixdur) + 1 == np.sum(
                filtered_raw[filtered_raw.imageno == image].is_saccade == 0)

        except AssertionError:
            print("image", image)
            print(np.sum(corpus_data[corpus_data.imageno == image].fixdur) + 1,
                  np.sum(filtered_raw[filtered_raw.imageno == image].is_saccade == 0))

        filtered_raw['identifier'] = filtered_raw['identifier'].astype(str)

    return filtered_raw