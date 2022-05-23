import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def import_data(file_name, folder=None):

    if folder:
        data_path = os.path.join(".", "data", folder, file_name)
    else:
        data_path = os.path.join(".", "data", file_name)

    df = pd.read_csv(data_path, low_memory=False)

    return df


def export_data(df, file_name, folder=None):

    if folder:
        csv_path = os.path.join(".", "data", folder, file_name + '.csv')
    else:
        csv_path = os.path.join(".", "data", file_name + '.csv')

    df.to_csv(csv_path, index=False)
    print("\nThe '{}' file has been printed in the '{}' folder.".format(
        csv_path.split('/')[-1], csv_path.split('/')[-2])
    )


def save_figure(figure_name):

    figure_path = os.path.join(".", "figure", figure_name + '.png')
    plt.savefig(figure_path)

    print(
        '\n' + figure_name.upper() + '\n',
        "The '{}' figure has been saved in the '{}' folder.\n"\
            .format(figure_path.split('/')[-1], figure_path.split('/')[-2])
        )


def calculate_confidence_interval_median(sample):

    medians = []
    for bootstrap_index in range(1000):
        resampled_sample = random.choices(sample, k=len(sample))
        medians.append(np.median(resampled_sample))

    return np.percentile(medians, 5), np.percentile(medians, 95)


def infer_repeat_offender_periods(strike_dates):

    overlapping_repeat_offender_periods = []

    # Define a repeat offender period every time there are two strikes in less than 90 days
    if len(strike_dates) > 1:
        for index in range(1, len(strike_dates)):
            if strike_dates[index] - strike_dates[index - 1] < np.timedelta64(90, 'D'):
                overlapping_repeat_offender_periods.append([
                    strike_dates[index],
                    strike_dates[index - 1] + np.timedelta64(90, 'D')
                ])

    # Because of the above definition, there can exist overlap between two periods
    # (e.g. the periods ['2021-02-23', '2021-03-17'] and ['2021-03-04', '2021-05-01'] are overlapping).
    # The script below merges the overlapping periods in one long period (['2021-02-23', '2021-05-01'])
    if len(overlapping_repeat_offender_periods) == 0:
        return []

    else:
        overlapping_repeat_offender_periods.sort(key=lambda interval: interval[0])
        repeat_offender_periods = [overlapping_repeat_offender_periods[0]]
        for current in overlapping_repeat_offender_periods:
            previous = repeat_offender_periods[-1]
            if current[0] <= previous[1]:
                previous[1] = max(previous[1], current[1])
            else:
                repeat_offender_periods.append(current)
        return repeat_offender_periods


def keep_repeat_offender_data(df, repeat_offender_periods):

    if len(repeat_offender_periods) == 0:
        return pd.DataFrame()

    repeat_offender_df_list = []
    for repeat_offender_period in repeat_offender_periods:
        new_df = df[(df['date'] >= repeat_offender_period[0]) &
                    (df['date'] <= repeat_offender_period[1])]
        if len(new_df) > 0:
            repeat_offender_df_list.append(new_df)

    if len(repeat_offender_df_list) > 0:
        return pd.concat(repeat_offender_df_list)
    else:
        return pd.DataFrame()


def keep_free_data(df, repeat_offender_periods):

    if len(repeat_offender_periods) == 0:
        return df

    free_df_list = []
    for ro_index in range(len(repeat_offender_periods) + 1):
        if ro_index == 0:
            new_df = df[df['date'] < repeat_offender_periods[0][0]]
        elif ro_index == len(repeat_offender_periods):
            new_df = df[df['date'] > repeat_offender_periods[-1][1]]
        else:
            new_df = df[(df['date'] > repeat_offender_periods[ro_index - 1][1]) &
                        (df['date'] < repeat_offender_periods[ro_index][0])]
        if len(new_df) > 0:
            free_df_list.append(new_df)

    if len(free_df_list) > 0:
        return pd.concat(free_df_list)
    else:
        return pd.DataFrame()
