import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


from constants import COLUMN_NAMES
from constants import MINIMUM_DISTANCE_BETWEEN_PEAKS


def read_data(file_path):
    """ 
    Read csv with data
    -------------------
    Return pandas DataFrame

    """
    raw_data = pd.read_csv(file_path, names=COLUMN_NAMES)
    raw_data['time'] = pd.to_datetime(raw_data.time_stamp).diff().fillna(
        pd.Timedelta(0)).dt.total_seconds().cumsum()

    return raw_data


def select_data_interval(data, start, end):
    """
    Select data between a interval
    -------------------------------
    return pandas DataFrame
    """
    mask = (data.time >= start) & (data.time <= end)
    return data[mask].reset_index(drop=True).copy()


def get_acceleration_peaks_index(data):
    """
    Calculate peaks in acceleration of all serie
    -------------------------------------
    return index of peak
    """
    peak_index, _ = find_peaks(
        data.ax, distance=MINIMUM_DISTANCE_BETWEEN_PEAKS, height=0)

    return peak_index


def get_entry_points_index(peaks, data):
    """
    Calculate the entry point of all serie
    ------------------------------------
    Return index of entry points
    """
    local_minimum_index = data.ax[(data.ax.shift(1) > data.ax) & (
        data.ax.shift(-1) > data.ax) & (data.ax < 0)].index
    local_minimum = data.iloc[local_minimum_index]

    entry_point_index = list()
    for index, row in peaks.iterrows():
        mask = (local_minimum.time < row.time)
        aux_df = local_minimum[mask]
        aux_df.sort_values(by='time')
        try:
            point_index = aux_df.tail(1).index.values[0]
        except IndexError:
            point_index = index

        entry_point_index.append(point_index)
    return entry_point_index


def get_peaks(data):
    """
    Calculate de peak points
    --------------------------
    Return pandas Dataframe with PEAK points
    """
    peaks_index = get_acceleration_peaks_index(data)
    return data.iloc[peaks_index].reset_index(drop=True)


def get_entry_points(data):
    """
    Calculate de entry points
    --------------------------
    Return pandas Dataframe with ENTRY points
    """
    peaks = get_peaks(data)
    entries_index = get_entry_points_index(peaks, data)
    return data.iloc[entries_index].reset_index(drop=True)


def get_strokes(data):
    """
    Calculate all strokes in a series
    -------------------------------------
    Return a Python list with a pandas DataFrame with stroke data
    """
    stroke_list = list()

    entries = get_entry_points(data)

    for index, row in entries.iterrows():
        if index == 0:
            mask = (data.time < row.time)
            aux_df = data[mask].copy()
            if not aux_df.empty:
                stroke_list.append(aux_df)
        else:
            mask = (data.time >=
                    entries.iloc[index - 1].time) & (data.time <= row.time)
            aux_df = data[mask].reset_index(drop=True).copy()
            stroke_list.append(aux_df)
    return stroke_list


def get_exit_and_air_points(data):
    """
    Calculate de exit point
    ---------------------------
    Return the point when there is a signal change in acceleration after PEAK
    (keep negative)
    """
    exits = data.copy()
    exits = exits.drop(index=exits.index)

    air_fase = data.copy()
    air_fase = air_fase.drop(index=air_fase.index)
    #i = 0

    strokes = get_strokes(data)

    for stroke in strokes:

        #print(f"Index: {i}")
        #i += 1

        # plt.figure(figsize=(20, 6))
        # plt.plot(stroke['time'], stroke.ax,
        #          label=r'Acceleration in x direction')
        # plt.xlabel('time (s)')
        # plt.ylabel(r'Acceleration (m/s²)')
        # plt.legend()
        # plt.show()

        index_max_ax = stroke.ax.argmax()
        time_max_acc = stroke.iloc[index_max_ax].time
        mask = (stroke.time > time_max_acc)
        aux_df = stroke[mask].copy()

        mask_signal_change = np.sign(aux_df.ax).diff().ne(0)
        exitpoint = aux_df[mask_signal_change].tail(1)

        exits = pd.concat([exits, exitpoint])
        exits = exits.sort_values(by='time', ascending=True)
        exits = exits.reset_index(drop=True)

        mask_air_fase = (aux_df.time >= exitpoint.time.values[0])
        aux_df_2 = aux_df[mask_air_fase].copy()

        # search for inflexion point with up concavity
        # if there isn't none get the mid point
        try:
            index_air_fase = aux_df_2.ax[(aux_df_2.ax.shift(1) > aux_df_2.ax) & (
                aux_df_2.ax.shift(-1) > aux_df_2.ax) & (aux_df_2.ax < 0)].index[0]
        except IndexError:
            print('AAA')
            mid = int(aux_df_2.shape[0]/2)
            index_air_fase = 5 + aux_df_2.head(1).index[0]

        air_point = stroke.iloc[[index_air_fase]]

        air_fase = pd.concat([air_fase, air_point])
        air_fase = air_fase.sort_values(by='time', ascending=True)
        air_fase = air_fase.reset_index(drop=True)

    return exits, air_fase


def indicators_calculator(data, fase, indicators_dict):

    indicators_dict[f'Speed Variation - {fase}'] = np.trapz(
        y=data.ax, x=data.time)

    indicators_dict[f'Pitch Amplitude - {fase}'] = data.pitch.max() - \
        data.pitch.min()

    indicators_dict[f'Mean Acceleration - {fase}'] = data.ax.mean()
    indicators_dict[f'Min Acceleration - {fase}'] = data.ax.min()
    indicators_dict[f'Max Acceleration - {fase}'] = data.ax.max()

    indicators_dict[f'Mean Pitch - {fase}'] = data.pitch.mean()

    indicators_dict[f'Useful Force - {fase}'] = (data.ax /
                                                 (data.ax + data.ay + data.az)).mean()

    # indicators_dict[f'Roll Amplitude - {fase}'] = data.roll.max() - \
    #     data.roll.min()

    # indicators_dict[f'Mean Roll - {fase}'] = data.roll.mean()


def delete_points_before_first_entry(strokes, entries, peaks, exits, air_fase):

    entry_time = entries.iloc[0].time
    peak_time = peaks.iloc[0].time
    exit_time = exits.iloc[0].time
    air_time = air_fase.iloc[0].time

    stroke_start_time = strokes[0].iloc[0].time

    # print(f"Entry: {entry_time}")
    # print(f"Peak: {peak_time}")
    # print(f"Exit: {exit_time}")
    # print(f"Air: {air_time}")

    if peak_time < entry_time:
        peaks.drop(0, inplace=True)
    if exit_time < entry_time:
        exits.drop(0, inplace=True)
    if air_time < entry_time:
        air_fase.drop(0, inplace=True)

    if stroke_start_time < entry_time:
        del strokes[0]

    return strokes, entries.reset_index(drop=True), peaks.reset_index(drop=True), exits.reset_index(drop=True), air_fase.reset_index(drop=True)


def strokes_indicators(data):
    entries = get_entry_points(data)
    peaks = get_peaks(data)
    exits, air_fase = get_exit_and_air_points(data)
    strokes = get_strokes(data)

    strokes, entries, peaks, exits, air_fase = delete_points_before_first_entry(
        strokes, entries, peaks, exits, air_fase)

    indicators = list()
    for stroke, (_, entry), (_, peak), (_, exit), (_, air) in zip(strokes, entries.iterrows(), peaks.iterrows(), exits.iterrows(), air_fase.iterrows()):

        indicators_dict = dict()

        entry_fase_mask = (stroke.time <= peak.time)
        entry_fase = stroke[entry_fase_mask].copy()
        indicators_calculator(entry_fase, "Entry Fase", indicators_dict)

        pull_fase_mask = (stroke.time >= peak.time) & (
            stroke.time <= exit.time)
        pull_fase = stroke[pull_fase_mask].copy()
        indicators_calculator(pull_fase, "Pull Fase", indicators_dict)

        exit_fase_mask = (stroke.time >= exit.time) & (stroke.time <= air.time)
        exit_fase = stroke[exit_fase_mask].copy()
        indicators_calculator(exit_fase, "Exit Fase", indicators_dict)

        air_fase_mask = (stroke.time >= air.time)
        air_fase = stroke[air_fase_mask].copy()
        indicators_calculator(air_fase, "Air Fase", indicators_dict)

        water_fase_mask = (stroke.time <= air.time)
        water_fase = stroke[water_fase_mask].copy()
        indicators_calculator(water_fase, "Water Fase", indicators_dict)

        indicators_dict['Water Time'] = air.time - \
            stroke.head(1).time.values[0]

        indicators_dict['Air Time'] = stroke.tail(1).time.values[0] - air.time

        indicators_dict['Stroke Rate'] = 60.0 / \
            (stroke.tail(1).time.values[0] - stroke.head(1).time.values[0])

        indicators_dict['Stroke Time'] = (stroke.tail(
            1).time.values[0] - stroke.head(1).time.values[0])

        indicators.append(indicators_dict)

    return pd.DataFrame(indicators)
