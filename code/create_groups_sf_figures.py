from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import scipy.stats as stats

from utils import (
    import_data, save_figure, export_data, 
    calculate_confidence_interval_median, infer_repeat_offender_periods,
    keep_repeat_offender_data, keep_free_data
)


pd.options.mode.chained_assignment = None
pd.set_option('use_inf_as_na', True)


def clean_df_url_posts(df_url_posts):

    df_url_posts = df_url_posts.dropna(subset=['url', 'account_id'])
    df_url_posts['account_id'] = df_url_posts['account_id'].astype(int)

    df_url_posts['datetime'] = pd.to_datetime(df_url_posts['datetime'])
    df_url_posts = df_url_posts.sort_values(by=['datetime'], ascending=True)
    df_url_posts = df_url_posts.drop_duplicates(subset=['url', 'account_id'])

    df_url_posts['date'] = pd.to_datetime(df_url_posts['date'])
    df_url_posts = df_url_posts[['url', 'account_id', 'date']]

    return df_url_posts


def infer_strike_dates(df_url_posts, df_url, account_id):

    df_url_posts_group = df_url_posts[df_url_posts['account_id'] == account_id]
    strike_dates = []

    for url in df_url_posts_group["url"].unique():
        first_post_date = df_url_posts_group[df_url_posts_group['url'] == url]['date'].values[0]
        fact_check_date = df_url[df_url['url_cleaned'] == url]['date'].values[0]
        strike_date = np.max([np.datetime64(first_post_date), np.datetime64(fact_check_date)])
        strike_dates.append(strike_date)

    strike_dates.sort()

    return strike_dates


def plot_example_sf_data_group(df_posts, df_url_posts, df_url, group_id):

    plt.figure(figsize=(8, 2.5))
    ax = plt.subplot(111)

    df_posts_group = df_posts[df_posts['account_id']==group_id]
    plt.title("The '" + df_posts_group['account_name'].values[0][:-3] + "' Facebook group")

    plt.plot(df_posts_group.groupby(by=["date"])['engagement'].mean(), color='royalblue')
    ax.set_ylabel("Engagement per post")
    plt.locator_params(axis='y', nbins=4)
    ax.grid(axis="y")

    strike_dates = infer_strike_dates(df_url_posts, df_url, group_id)
    for date in strike_dates:
        plt.plot([date, date], [-3, -0.3], color='C3')
    plt.text(
        s='Known strikes', color='C3', fontweight='bold',
        x=np.datetime64('2021-07-15'), horizontalalignment='right', 
        y=-1, verticalalignment='top'
    )

    repeat_offender_periods = infer_repeat_offender_periods(strike_dates)
    for period in repeat_offender_periods:
        plt.axvspan(period[0], period[1], ymin=1/11, facecolor='C3', alpha=0.1)
    patch1 = mpatches.Patch(facecolor='pink', alpha=0.4, edgecolor='k')
    patch2 = mpatches.Patch(facecolor='white', alpha=0.4, edgecolor='k')
    legend = plt.legend([patch1, patch2], 
        ["'Repeat offender' periods\n(2 strikes in less than 90 days)", "'Normal' periods"],
        loc='upper right', framealpha=1
    )
    legend.get_patches()[0].set_y(6)

    plt.text(
        np.datetime64('2021-07-10'), 13, 'Percentage change = -30%', 
        color='C3', ha='center', va='center'
    )

    plt.xlim(np.datetime64('2021-01-01'), np.datetime64('2021-12-15'))
    plt.xticks([
        np.datetime64('2021-01-01'), np.datetime64('2021-06-01'), 
        np.datetime64('2021-11-01')
    ])
    ax.tick_params(axis='x', which='both', length=0)
    plt.ylim(-3, 30)
    ax.set_frame_on(False)
    plt.tight_layout()
    save_figure('figure_3_top')


def calculate_engagement_percentage_change(df_posts, df_url_posts, df_url):

    sumup_df = pd.DataFrame(columns=[
        'group_id',
        'group_name',
        'group_url', 
        'engagement_normal', 
        'engagement_repeat'
    ])

    for group_id in df_posts['account_id'].unique():

        df_posts_group = df_posts[df_posts["account_id"] == group_id]
        group_name = df_posts_group['account_name'].unique()[0]
        group_url = df_posts_group['account_url'].unique()[0]

        strike_dates = infer_strike_dates(df_url_posts, df_url, group_id)
        repeat_offender_periods = infer_repeat_offender_periods(strike_dates)

        normal_df = keep_free_data(df_posts_group, repeat_offender_periods)
        repeat_df = keep_repeat_offender_data(df_posts_group, repeat_offender_periods)

        if (len(normal_df) > 0) & (len(repeat_df) > 0):
            
            sumup_df = sumup_df.append({
                'group_id': group_id,
                'group_name': group_name, 
                'group_url': group_url,
                'engagement_normal': np.mean(normal_df['engagement']),
                'engagement_repeat': np.mean(repeat_df['engagement']),
            }, ignore_index=True)
            
    sumup_df['percentage_change_engagement'] = ((sumup_df['engagement_repeat'] - sumup_df['engagement_normal'])/
                                                sumup_df['engagement_normal']) * 100
    sumup_df = sumup_df.dropna()
    sumup_df['repeat_vs_free_percentage_change'] = sumup_df['percentage_change_engagement'].apply(lambda x: str(int(np.round(x))) + '%')

    return sumup_df


def print_statistics(sumup_df):

    print("Engagement percentage change for the 'Build The Wall' group:",
        sumup_df[sumup_df['group_id']==1869250196644008]['percentage_change_engagement'].values[0]
    )

    print('\nSample size:', len(sumup_df))

    print('Median engagement per post normal:', np.median(sumup_df['engagement_normal']))
    print('Median engagement per post repeat:', np.median(sumup_df['engagement_repeat']))

    print('Median engagement percentage change:', np.median(sumup_df['percentage_change_engagement']))
    w, p = stats.wilcoxon(sumup_df['percentage_change_engagement'])
    print('Wilcoxon test against zero: w =', w, ', p =', p)


def plot_engagement_change(sumup_df):

    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]})
    fig.set_size_inches(8, 3)

    # Median engagements before or free VS after or repeat
    ax0.bar(
        [0.1], [np.median(sumup_df['engagement_normal'])],
        color='white', edgecolor='black', width=0.8, alpha=0.4
    )
    ax0.bar(
        [0.9], [np.median(sumup_df['engagement_repeat'])],
        color='pink', edgecolor='black', width=0.8, alpha=0.4
    )
    low_normal, high_normal = calculate_confidence_interval_median(sumup_df['engagement_normal'].values)
    ax0.plot([0.1, 0.1], [low_normal, high_normal], color='black', linewidth=0.9)
    low_repeat, high_repeat = calculate_confidence_interval_median(sumup_df['engagement_repeat'].values)
    ax0.plot([0.9, 0.9], [low_repeat, high_repeat], color='black', linewidth=0.9)
    ax0.set_xticks([-0.08, 1.08], ["'Normal'\nperiods", "'Repeat\noffender'\nperiods"])
    ax0.tick_params(axis='x', which='both', length=0)
    ax0.set_xlim(-.5, 1.5)
    ax0.set_ylabel('Median engagement per post')
    ax0.set_ylim(-.1, 11.5)
    ax0.set_frame_on(False)

    # Percentage change in engagement
    random_y = list(np.random.random(len(sumup_df)))
    ax1.plot(sumup_df['percentage_change_engagement'].values, random_y, 
             'o', markerfacecolor='royalblue', markeredgecolor='blue', alpha=0.6,
             label='Facebook pages')

    low, high = calculate_confidence_interval_median(sumup_df['percentage_change_engagement'].values)
    ax1.plot([low, np.median(sumup_df['percentage_change_engagement']), high], 
             [0.5 for x in range(3)], '|-', color='navy', 
             linewidth=2, markersize=12, markeredgewidth=2)
        
    ax1.set_xlabel('Percentage change in engagement\nbetween the repeat offender and normal periods')
    ax1.set_xlim(-110, 135)
    ax1.set_ylim(-.2, 1.2)

    ax1.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xticks(
        [-100, -75, -50, -25, 0, 25, 50, 75, 100, 125], 
        ['-100%', '-75%', '-50%', '-25%', ' 0%', '+25%', '+50%', '+75%', '+100%', '+125%']
    )
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    plt.title("{} misinformation Facebook groups (Science Feedback data)".format(len(sumup_df)), loc='left')

    fig.tight_layout()
    save_figure('figure_3_bottom')


if __name__=="__main__":

    ### Figure 3 ###

    df_posts = import_data('posts_smaller.csv', folder='groups_sciencefeedback_data')
    df_posts['date'] = pd.to_datetime(df_posts['date'])

    df_url = import_data('appearances_2021-12-15.csv', folder='groups_sciencefeedback_data')
    df_url['date'] = pd.to_datetime(df_url['date'])

    df_url_posts = import_data('posts_url_2021-12-15.csv', folder='groups_sciencefeedback_data')
    df_url_posts = clean_df_url_posts(df_url_posts)

    plot_example_sf_data_group(df_posts, df_url_posts, df_url, group_id=1869250196644008)
    
    sumup_df = calculate_engagement_percentage_change(df_posts, df_url_posts, df_url)
    print_statistics(sumup_df)

    plot_engagement_change(sumup_df)
    # export_data(sumup_df[['group_url', 'repeat_vs_free_percentage_change']], 'list_groups_sciencefeedback', 'groups_sciencefeedback_data')
