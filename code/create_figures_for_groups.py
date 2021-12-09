from datetime import datetime, timedelta
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats


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


def clean_df_posts(df_posts):

    df_posts['date'] = pd.to_datetime(df_posts['date'])
    
    df_posts["share"]   = df_posts[["actual_share_count"]].astype(int)
    df_posts["comment"] = df_posts[["actual_comment_count"]].astype(int)
    df_posts["reaction"] = df_posts[["actual_like_count", "actual_favorite_count", "actual_love_count",
        "actual_wow_count", "actual_haha_count", "actual_sad_count",
        "actual_angry_count", "actual_thankful_count"]].sum(axis=1).astype(int)

    df_posts['engagement'] = df_posts[["share", "comment", "reaction"]].sum(axis=1)

    return df_posts


def calculate_engagement_percentage_change(df_posts, df_dates, period_length=30):

    sumup_df = pd.DataFrame(columns=[
        'account_id',
        'group_name', 
        'engagement_before', 
        'engagement_after'
    ])

    for group_id in df_posts['account_id'].unique():

        group_name = df_posts[df_posts['account_id']==group_id]['account_name'].iloc[0]
        reduced_distribution_date = df_dates[df_dates['group_id']==group_id]['post_date'].values[0]
        reduced_distribution_date = datetime.strptime(str(reduced_distribution_date)[:10], '%Y-%m-%d')
        df_posts_group = df_posts[df_posts["account_id"] == group_id]

        posts_df_group_before = df_posts_group[
            (df_posts_group['date'] >= reduced_distribution_date - timedelta(days=period_length)) &
            (df_posts_group['date'] < reduced_distribution_date)
        ]
        posts_df_group_after = df_posts_group[
            (df_posts_group['date'] > reduced_distribution_date) &
            (df_posts_group['date'] <= reduced_distribution_date + timedelta(days=period_length))
        ]

        if (len(posts_df_group_before) > 0) & (len(posts_df_group_after) > 0):
            
            sumup_df = sumup_df.append({
                'group_id': group_id,
                'group_name': group_name, 
                'engagement_before': np.mean(posts_df_group_before['engagement']),
                'engagement_after': np.mean(posts_df_group_after['engagement']),
            }, ignore_index=True)
            
    sumup_df['percentage_change_engagement'] = ((sumup_df['engagement_after'] - sumup_df['engagement_before'])/
                                                sumup_df['engagement_before']) * 100
    return sumup_df


def calculate_confidence_interval_median(sample):

    medians = []
    for bootstrap_index in range(1000):
        resampled_sample = random.choices(sample, k=len(sample))
        medians.append(np.median(resampled_sample))

    return np.percentile(medians, 5), np.percentile(medians, 95)


def plot_engagement_percentage_change(sumup_df):

    plt.figure(figsize=(7, 3))
    ax = plt.subplot()
    plt.title("{} 'reduced distribution' Facebook groups".format(len(sumup_df)))

    random_y = list(np.random.random(len(sumup_df)))
    plt.plot(sumup_df['percentage_change_engagement'].values, 
             random_y, 
             'o', markerfacecolor='royalblue', markeredgecolor='blue', alpha=0.6,
             label='Facebook pages')
    plt.xlabel("Engagement percentage change\nafter the 'reduced distribution' start date", size='large')

    low, high = calculate_confidence_interval_median(sumup_df['percentage_change_engagement'].values)
    plt.plot([low, np.median(sumup_df['percentage_change_engagement']), high], 
             [0.5 for x in range(3)], '|-', color='navy', 
             linewidth=2, markersize=12, markeredgewidth=2)

    plt.xlim(-120, 135)
    plt.ylim(-.2, 1.2)

    plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
    plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100, 125], 
            ['-100%', '-75%', '-50%', '-25%', ' 0%', '+25%', '+50%', '+75%', '+100%', '+125%'])
    plt.yticks([])
    
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.savefig('./figure/engagement_percentage_change_for_self_declared_reduced_groups.png')


if __name__=="__main__":

    df_dates = pd.read_csv('./data/manually_filtered_reduced_posts.csv')
    df_dates['post_date'] = pd.to_datetime(df_dates['post_date'])
    plot_reduced_posts_dates(df_dates)

    df_dates = df_dates.sort_values(by=['url_group', 'post_date']).drop_duplicates(subset=['url_group'])
    df_dates['group_id'] = df_dates['url_group'].apply(lambda x: x.split('/')[-1]).astype(int)

    df_posts = pd.read_csv('./data/posts_reduced_groups_2021-12-02.csv')
    df_posts = clean_df_posts(df_posts)

    sumup_df = calculate_engagement_percentage_change(df_posts, df_dates)
    plot_engagement_percentage_change(sumup_df)
    print('Number of groups:', len(sumup_df))
    print('Median of the engagement percentage changes for the self-declared reduced groups:', 
          np.median(sumup_df['percentage_change_engagement']))
    w, p = stats.wilcoxon(sumup_df['percentage_change_engagement'])
    print('Wilcoxon test against zero: w =', w, ', p =', p)