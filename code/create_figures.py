from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_reduced_posts_dates(df):

    plt.figure(figsize=(7, 1.5))
    ax = plt.subplot(111)

    for date in df['post_date'].values:
        plt.axvline(x=date, color=(0, 0, 0, 0.8), linewidth=1)

    plt.xlim(
        np.datetime64(datetime.strptime('2020-06-01', '%Y-%m-%d')), 
        np.datetime64(datetime.strptime('2021-11-30', '%Y-%m-%d'))
    )
    plt.xticks(
        ticks=[np.datetime64(x) for x in ['2020-07-01', '2020-10-01', '2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01']], 
        labels=['Jul 2020', 'Oct 2020', 'Jan 2021', 'Apr 2021', 'Jul 2021', 'Oct 2021']
    )
    plt.xlabel("Dates of the posts sharing a message or a screenshot saying\n'Your group's distribution is reduced due to false information'", size='large')
    plt.yticks([])

    ax.set_frame_on(False)
    plt.tight_layout()
    plt.savefig('./figure/reduced_posts_dates.png')


if __name__=="__main__":
    df = pd.read_csv('./data/manually_filtered_reduced_posts.csv')
    df['post_date'] = pd.to_datetime(df['post_date'])
    plot_reduced_posts_dates(df)
