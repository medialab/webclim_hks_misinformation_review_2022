from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import scipy.stats as stats

from utils import import_data, save_figure, export_data, calculate_confidence_interval_median


pd.options.mode.chained_assignment = None
pd.set_option('use_inf_as_na', True)


def plot_reduced_posts_dates(df_dates):

    plt.figure(figsize=(8, 1.3))
    ax = plt.subplot(111)

    for date in df_dates['post_date'].values:
        plt.axvline(x=date, color=(0, 0, 0, 0.8), linewidth=1)

    plt.xlim(
        np.datetime64(datetime.strptime('2020-06-01', '%Y-%m-%d')), 
        np.datetime64(datetime.strptime('2021-11-30', '%Y-%m-%d'))
    )
    plt.xticks(
        ticks=[np.datetime64(x) for x in ['2020-07-01', '2020-10-01', '2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01']], 
        labels=['Jul 2020', 'Oct 2020', 'Jan 2021', 'Apr 2021', 'Jul 2021', 'Oct 2021']
    )
    plt.xlabel("Dates of the posts sharing a message or a screenshot saying\n'Your group's distribution is reduced due to false information'")
    plt.yticks([])

    ax.set_frame_on(False)
    plt.tight_layout()
    save_figure('figure_1_bottom')


def plot_example_ct_search_group(df_posts, df_dates, group_id):

    plt.figure(figsize=(8, 2.5))
    ax = plt.subplot(111)

    df_posts_group = df_posts[df_posts['account_id']==group_id]
    plt.title("The '" + df_posts_group['account_name'].values[0] + "' Facebook group")

    plt.plot(df_posts_group.groupby(by=["date"])['engagement'].mean(), color='royalblue')
    ax.set_ylabel("Engagement per post")
    plt.locator_params(axis='y', nbins=4)
    ax.grid(axis="y")

    reduced_date = df_dates[df_dates['group_id']==group_id]['post_date'].values[0]
    plt.axvline(x=reduced_date, color='C3', linestyle='--')
    plt.xticks([
        np.datetime64('2021-01-01'), np.datetime64('2021-07-01'), 
        np.datetime64('2021-12-31'), np.datetime64(reduced_date)
    ])
    plt.gca().get_xticklabels()[-1].set_color('C3')
    ax.tick_params(axis='x', which='both', length=0)

    plt.axvspan(reduced_date - np.timedelta64(30, 'D'), reduced_date, facecolor='springgreen', alpha=0.2)
    plt.axvspan(reduced_date, reduced_date + np.timedelta64(30, 'D'), facecolor='gold', alpha=0.2)
    patch1 = mpatches.Patch(facecolor='springgreen', alpha=0.4, edgecolor='k')
    patch2 = mpatches.Patch(facecolor='gold', alpha=0.4, edgecolor='k')
    plt.legend(
        [patch1, patch2], 
        ['30-day period before the notification date', '30-day period after the notification date'],
        loc='upper right', framealpha=1
    )
    plt.text(
        reduced_date + np.timedelta64(25, 'D'), 13, 'Percentage\nchange =\n-29%', 
        color='C3', ha='center', va='center'
    )

    plt.xlim(np.datetime64('2021-01-01'), np.datetime64('2021-12-31'))
    plt.ylim(-1, 21)
    ax.set_frame_on(False)
    plt.tight_layout()
    save_figure('figure_2_top')


def calculate_engagement_percentage_change(df_posts, df_dates):

    sumup_df = pd.DataFrame(columns=[
        'group_id',
        'group_name',
        'group_url', 
        'engagement_before', 
        'engagement_after'
    ])

    for group_id in df_posts['account_id'].unique():

        df_posts_group = df_posts[df_posts["account_id"] == group_id]
        group_name = df_posts_group['account_name'].unique()[0]
        group_url = df_posts_group['account_url'].unique()[0]

        reduced_distribution_date = df_dates[df_dates['group_id']==group_id]['post_date'].values[0]
        reduced_distribution_date = datetime.strptime(str(reduced_distribution_date)[:10], '%Y-%m-%d')

        before_df = df_posts_group[
            (df_posts_group['date'] >= reduced_distribution_date - timedelta(days=30)) &
            (df_posts_group['date'] < reduced_distribution_date)
        ]
        after_df = df_posts_group[
            (df_posts_group['date'] > reduced_distribution_date) &
            (df_posts_group['date'] <= reduced_distribution_date + timedelta(days=30))
        ]

        if (len(before_df) > 0) & (len(after_df) > 0):
            
            sumup_df = sumup_df.append({
                'group_id': group_id,
                'group_name': group_name, 
                'group_url': group_url,
                'engagement_before': np.mean(before_df['engagement']),
                'engagement_after': np.mean(after_df['engagement']),
            }, ignore_index=True)
            
    sumup_df['percentage_change_engagement'] = ((sumup_df['engagement_after'] - sumup_df['engagement_before'])/
                                                sumup_df['engagement_before']) * 100
    sumup_df = sumup_df.dropna()
    sumup_df['after_vs_before_percentage_change'] = sumup_df['percentage_change_engagement'].apply(lambda x: str(int(np.round(x))) + '%')

    return sumup_df


def print_statistics(sumup_df):

    print("Engagement percentage change for the 'Woke Shift' group:",
        sumup_df[sumup_df['group_id']==3332382583652839]['percentage_change_engagement'].values[0]
    )

    print('\nSample size:', len(sumup_df))

    print('Median engagement per post before:', np.median(sumup_df['engagement_before']))
    print('Median engagement per post after:', np.median(sumup_df['engagement_after']))

    print('Median engagement percentage change:', np.median(sumup_df['percentage_change_engagement']))
    w, p = stats.wilcoxon(sumup_df['percentage_change_engagement'])
    print('Wilcoxon test against zero: w =', w, ', p =', p)


def plot_engagement_change(sumup_df):

    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]})
    fig.set_size_inches(8, 2.5)

    # Median engagements before or free VS after or repeat
    ax0.bar(
        [0.1], [np.median(sumup_df['engagement_before'])],
        color='springgreen', edgecolor='black', width=0.8, alpha=0.4
    )
    ax0.bar(
        [0.9], [np.median(sumup_df['engagement_after'])],
        color='gold', edgecolor='black', width=0.8, alpha=0.4
    )
    low_before, high_before = calculate_confidence_interval_median(sumup_df['engagement_before'].values)
    ax0.plot([0.1, 0.1], [low_before, high_before], color='black', linewidth=0.9)
    low_after, high_after = calculate_confidence_interval_median(sumup_df['engagement_after'].values)
    ax0.plot([0.9, 0.9], [low_after, high_after], color='black', linewidth=0.9)
    ax0.set_xticks([0, 1], ['30 days\nbefore', '30 days\nafter'])
    ax0.set_xlabel('the notification date')
    ax0.tick_params(axis='x', which='both', length=0)
    ax0.set_xlim(-.5, 1.5)
    ax0.set_ylabel('Median engagement per post')
    ax0.set_ylim(-.1, 10.5)
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
        
    ax1.set_xlabel('Percentage change in engagement\n30 days after minus before the notification date')
    ax1.set_xlim(-110, 135)
    ax1.set_ylim(-.1, 1.1)

    ax1.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xticks(
        [-100, -75, -50, -25, 0, 25, 50, 75, 100, 125], 
        ['-100%', '-75%', '-50%', '-25%', ' 0%', '+25%', '+50%', '+75%', '+100%', '+125%']
    )
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    plt.title("{} misinformation Facebook groups (CrowdTangle search)".format(len(sumup_df)), loc='left')

    fig.tight_layout()
    save_figure('figure_2_bottom')


if __name__=="__main__":

    ### Figure 1 ###

    df_dates = import_data('manually_filtered_reduced_posts.csv', folder='groups_crowdtangle_search')
    df_dates['post_date'] = pd.to_datetime(df_dates['post_date'])
    plot_reduced_posts_dates(df_dates)

    ### Figure 2 ###

    df_dates = df_dates.sort_values(by=['group_url', 'post_date']).drop_duplicates(subset=['group_url'])
    df_dates['group_id'] = df_dates['group_url'].apply(lambda x: x.split('/')[-1]).astype(int)

    df_posts = import_data('posts_smaller.csv', folder='groups_crowdtangle_search')
    df_posts['date'] = pd.to_datetime(df_posts['date'])

    plot_example_ct_search_group(df_posts, df_dates, group_id=3332382583652839)

    sumup_df = calculate_engagement_percentage_change(df_posts, df_dates)
    print_statistics(sumup_df)
    plot_engagement_change(sumup_df)
    export_data(sumup_df[['group_url', 'after_vs_before_percentage_change']], 'list_groups_crowdtangle_search', 'groups_crowdtangle_search')
