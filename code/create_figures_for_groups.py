from datetime import datetime, timedelta
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

pd.options.mode.chained_assignment = None
pd.set_option('use_inf_as_na', True)


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


def get_before_and_after_data(df_posts_group, df_dates, group_id):

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

    return before_df, after_df


def calculate_engagement_percentage_change(df_posts, how='30_days_before_vs_after', df_dates=None, df_url_posts=None, df_url=None):

    sumup_df = pd.DataFrame(columns=[
        'account_id',
        'group_name', 
        'engagement_before', 
        'engagement_after'
    ])

    for group_id in df_posts['account_id'].unique():

        df_posts_group = df_posts[df_posts["account_id"] == group_id]
        group_name = df_posts_group['account_name'].unique()[0]

        if how == '30_days_before_vs_after':
            before_df, after_df = get_before_and_after_data(df_posts_group, df_dates, group_id)
        elif how == 'repeat_offender_vs_free_periods':
            before_df, after_df = get_repeat_and_free_data(df_posts_group, group_id, df_url_posts, df_url)

        if (len(before_df) > 0) & (len(after_df) > 0):
            
            sumup_df = sumup_df.append({
                'group_id': group_id,
                'group_name': group_name, 
                'engagement_before': np.mean(before_df['engagement']),
                'engagement_after': np.mean(after_df['engagement']),
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


def plot_engagement_percentage_change(sumup_df, style):

    if style == 'figure_3':
        plt.figure(figsize=(7, 2.5))
        plt.title("{} 'reduced distribution' Facebook groups (CrowdTangle search)".format(len(sumup_df)))
        plt.xlabel("Engagement percentage change\nafter the 'reduced distribution' start date", size='large')
    elif style == 'figure_4':
        plt.figure(figsize=(7, 3))
        plt.title("{} 'repeat offender' Facebook groups (Science Feedback data)".format(len(sumup_df)))
        plt.xlabel("Engagement percentage change\nbetween the 'repeat offender' and 'normal' periods", size='large')

    ax = plt.subplot()

    random_y = list(np.random.random(len(sumup_df)))
    plt.plot(sumup_df['percentage_change_engagement'].values, random_y, 
             'o', markerfacecolor='royalblue', markeredgecolor='blue', alpha=0.6,
             label='Facebook pages')

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

    if style == 'figure_3':
        plt.savefig('./figure/engagement_percentage_change_for_self_declared_reduced_groups.png')
    elif style == 'figure_4':
        plt.savefig('./figure/engagement_percentage_change_for_repeat_offender_groups_sf.png')


def print_statistics(sumup_df):

    print('Sample size:', len(sumup_df))
    print('Median engagement percentage change:', np.median(sumup_df['percentage_change_engagement']))
    w, p = stats.wilcoxon(sumup_df['percentage_change_engagement'])
    print('Wilcoxon test against zero: w =', w, ', p =', p)


def clean_df_url_posts(df_url_posts):

    df_url_posts = df_url_posts.dropna(subset=['url', 'account_id'])
    df_url_posts['account_id'] = df_url_posts['account_id'].astype(int)

    df_url_posts['datetime'] = pd.to_datetime(df_url_posts['datetime'])
    df_url_posts = df_url_posts.sort_values(by=['datetime'], ascending=True)
    df_url_posts = df_url_posts.drop_duplicates(subset=['url', 'account_id'])

    df_url_posts['date'] = pd.to_datetime(df_url_posts['date'])
    df_url_posts = df_url_posts[['url', 'account_id', 'date']]

    return df_url_posts


def infer_strike_dates_for_groups(df_url_posts, df_url, account_id):

    df_url_posts_group = df_url_posts[df_url_posts['account_id'] == account_id]
    strike_dates = []

    for url in df_url_posts_group["url"].unique():
        first_post_date = df_url_posts_group[df_url_posts_group['url'] == url]['date'].values[0]
        fact_check_date = df_url[df_url['url_cleaned'] == url]['date'].values[0]
        strike_date = np.max([np.datetime64(first_post_date), np.datetime64(fact_check_date)])
        strike_dates.append(strike_date)

    strike_dates.sort()

    return strike_dates


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


def keep_repeat_offender_posts(df_posts_group, repeat_offender_periods):
    
    if len(repeat_offender_periods) == 0:
        return pd.DataFrame()

    repeat_offender_df_list = []
    for repeat_offender_period in repeat_offender_periods:
        new_df = df_posts_group[(df_posts_group['date'] >= repeat_offender_period[0]) &
                                (df_posts_group['date'] <= repeat_offender_period[1])]
        if len(new_df) > 0:
            repeat_offender_df_list.append(new_df)
    
    if len(repeat_offender_df_list) > 0:
        return pd.concat(repeat_offender_df_list)
    else:
        return pd.DataFrame()


def keep_free_posts(df_posts_group, repeat_offender_periods):
        
    if len(repeat_offender_periods) == 0:
        return df_posts_group

    free_df_list = []
    for ro_index in range(len(repeat_offender_periods) + 1):
        if ro_index == 0:
            new_df = df_posts_group[df_posts_group['date'] < repeat_offender_periods[0][0]]
        elif ro_index == len(repeat_offender_periods):
            new_df = df_posts_group[df_posts_group['date'] > repeat_offender_periods[-1][1]]
        else:
            new_df = df_posts_group[(df_posts_group['date'] > repeat_offender_periods[ro_index - 1][1]) &
                                    (df_posts_group['date'] < repeat_offender_periods[ro_index][0])]
        if len(new_df) > 0:
            free_df_list.append(new_df)
    
    if len(free_df_list) > 0:
        return pd.concat(free_df_list)
    else:
        return pd.DataFrame()


def get_repeat_and_free_data(df_posts_group, group_id, df_url_posts, df_url):

    strike_dates = infer_strike_dates_for_groups(df_url_posts, df_url, group_id)
    repeat_offender_periods = infer_repeat_offender_periods(strike_dates)

    free_df = keep_free_posts(df_posts_group, repeat_offender_periods)
    repeat_offender_df = keep_repeat_offender_posts(df_posts_group, repeat_offender_periods)

    return free_df, repeat_offender_df


if __name__=="__main__":


    ### Figure 2 ###

    df_dates = pd.read_csv('./data/manually_filtered_reduced_posts.csv')
    df_dates['post_date'] = pd.to_datetime(df_dates['post_date'])
    plot_reduced_posts_dates(df_dates)

    ### Figure 3 ###

    df_dates = df_dates.sort_values(by=['url_group', 'post_date']).drop_duplicates(subset=['url_group'])
    df_dates['group_id'] = df_dates['url_group'].apply(lambda x: x.split('/')[-1]).astype(int)

    df_posts_1 = pd.read_csv('./data/posts_reduced_groups_2022-01-03.csv')
    df_posts_1 = clean_df_posts(df_posts_1)

    sumup_df_1 = calculate_engagement_percentage_change(df_posts=df_posts_1, how='30_days_before_vs_after', df_dates=df_dates)
    plot_engagement_percentage_change(sumup_df_1, style='figure_3')
    print_statistics(sumup_df_1)
 
    ### Figure 4 ###

    df_posts_2 = pd.read_csv('./data/posts_repeat_offender_groups_2021-12-15.csv')
    df_posts_2 = df_posts_2[~df_posts_2['account_id'].isin(df_posts_1['account_id'].unique())]
    df_posts_2 = clean_df_posts(df_posts_2)

    df_url = pd.read_csv('./data/appearances_2021-12-15.csv')
    df_url['date'] = pd.to_datetime(df_url['date'])

    df_url_posts = pd.read_csv('./data/posts_url_2021-12-15.csv')
    df_url_posts = clean_df_url_posts(df_url_posts)

    sumup_df_2 = calculate_engagement_percentage_change(df_posts=df_posts_2, how='repeat_offender_vs_free_periods', 
                                                        df_url_posts=df_url_posts, df_url=df_url)
    plot_engagement_percentage_change(sumup_df_2, style='figure_4')
    print_statistics(sumup_df_2)
