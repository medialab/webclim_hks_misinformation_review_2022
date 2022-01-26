from datetime import datetime, timedelta
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

from utils import import_data, save_figure, export_data


pd.options.mode.chained_assignment = None
pd.set_option('use_inf_as_na', True)


def clean_buzzsumo_data(df):

    df['date'] = [datetime.fromtimestamp(x).date() for x in df['published_date']]
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['url'])
        
    return df


def infer_strike_dates_for_domains(df_url, domain):
    strike_dates = list(df_url[df_url['domain']==domain]['date'].values)
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


def calculate_engagement_percentage_change_for_domains(df, df_url):

    sumup_df = pd.DataFrame(columns=[
        'domain_name',
        'engagement_repeat', 
        'engagement_free'
    ])

    for domain in df['domain_name'].unique():

        df_domain = df[df['domain_name'] == domain]

        strike_dates = infer_strike_dates_for_domains(df_url, domain)
        repeat_offender_periods = infer_repeat_offender_periods(strike_dates)

        free_df = keep_free_data(df_domain, repeat_offender_periods)
        repeat_offender_df = keep_repeat_offender_data(df_domain, repeat_offender_periods)

        if (len(repeat_offender_df) > 0) & (len(free_df) > 0): 
            sumup_df = sumup_df.append({
                'domain_name': domain,
                'engagement_repeat': np.mean(repeat_offender_df['total_facebook_shares']),
                'engagement_free': np.mean(free_df['total_facebook_shares']),
            }, ignore_index=True)
            
    sumup_df['percentage_change_engagement'] = ((sumup_df['engagement_repeat'] - sumup_df['engagement_free'])/
                                                sumup_df['engagement_free']) * 100
    sumup_df = sumup_df.dropna()
    sumup_df['repeat_vs_free_percentage_change'] = sumup_df['percentage_change_engagement'].apply(lambda x: str(int(np.round(x))) + '%')

    return sumup_df


def calculate_confidence_interval_median(sample):

    medians = []
    for bootstrap_index in range(1000):
        resampled_sample = random.choices(sample, k=len(sample))
        medians.append(np.median(resampled_sample))

    return np.percentile(medians, 5), np.percentile(medians, 95)


def plot_engagement_percentage_change(sumup_df, figure_name):

    if figure_name == 'figure_1':
        plt.figure(figsize=(7, 3.5))
        plt.title("{} misinformation websites (Condor data)".format(len(sumup_df)))
    elif figure_name == 'figure_2':
        plt.figure(figsize=(7, 3))
        plt.title("{} misinformation websites (Science Feedback data)".format(len(sumup_df)))
    elif figure_name == 'figure_4':
        plt.figure(figsize=(7, 2.5))
        plt.title("{} misinformation Facebook groups (CrowdTangle search)".format(len(sumup_df)))
    elif figure_name == 'figure_5':
        plt.figure(figsize=(7, 3))
        plt.title("{} misinformation Facebook groups (Science Feedback data)".format(len(sumup_df)))

    ax = plt.subplot()

    random_y = list(np.random.random(len(sumup_df)))
    plt.plot(sumup_df['percentage_change_engagement'].values, random_y, 
             'o', markerfacecolor='royalblue', markeredgecolor='blue', alpha=0.6,
             label='Facebook pages')

    low, high = calculate_confidence_interval_median(sumup_df['percentage_change_engagement'].values)
    plt.plot([low, np.median(sumup_df['percentage_change_engagement']), high], 
             [0.5 for x in range(3)], '|-', color='navy', 
             linewidth=2, markersize=12, markeredgewidth=2)

    if figure_name == 'figure_4':
        plt.xlabel("Engagement percentage change\nafter the reduced distribution start date", size='large')
    else:
        plt.xlabel("Engagement percentage change\nbetween the repeat offender and normal periods", size='large')
    plt.xlim(-120, 135)
    plt.ylim(-.2, 1.2)

    plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
    plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100, 125], 
            ['-100%', '-75%', '-50%', '-25%', ' 0%', '+25%', '+50%', '+75%', '+100%', '+125%'])
    plt.yticks([])
    
    ax.set_frame_on(False)
    plt.tight_layout()
    save_figure(figure_name)


def print_statistics(sumup_df):

    print('Sample size:', len(sumup_df))
    print('Median engagement percentage change:', np.median(sumup_df['percentage_change_engagement']))
    w, p = stats.wilcoxon(sumup_df['percentage_change_engagement'])
    print('Wilcoxon test against zero: w =', w, ', p =', p)


def clean_crowdtangle_data(df_posts):

    df_posts['date'] = pd.to_datetime(df_posts['date'])
    
    df_posts["share"]   = df_posts[["actual_share_count"]].astype(int)
    df_posts["comment"] = df_posts[["actual_comment_count"]].astype(int)
    df_posts["reaction"] = df_posts[["actual_like_count", "actual_favorite_count", "actual_love_count",
        "actual_wow_count", "actual_haha_count", "actual_sad_count",
        "actual_angry_count", "actual_thankful_count"]].sum(axis=1).astype(int)

    df_posts['engagement'] = df_posts[["share", "comment", "reaction"]].sum(axis=1)

    return df_posts


def plot_reduced_posts_dates(df, figure_name):

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
    save_figure(figure_name)


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


def calculate_engagement_percentage_change_for_groups(df_posts, how='30_days_before_vs_after', df_dates=None, df_url_posts=None, df_url=None):

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

        if how == '30_days_before_vs_after':

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
        elif how == 'repeat_offender_vs_free_periods':

            strike_dates = infer_strike_dates_for_groups(df_url_posts, df_url, group_id)
            repeat_offender_periods = infer_repeat_offender_periods(strike_dates)

            before_df = keep_free_data(df_posts_group, repeat_offender_periods)
            after_df = keep_repeat_offender_data(df_posts_group, repeat_offender_periods)

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
    if how == '30_days_before_vs_after':
        sumup_df['after_vs_before_percentage_change'] = sumup_df['percentage_change_engagement'].apply(lambda x: str(int(np.round(x))) + '%')
    elif how == 'repeat_offender_vs_free_periods':
        sumup_df['repeat_vs_free_percentage_change'] = sumup_df['percentage_change_engagement'].apply(lambda x: str(int(np.round(x))) + '%')

    return sumup_df


def clean_df_url_posts(df_url_posts):

    df_url_posts = df_url_posts.dropna(subset=['url', 'account_id'])
    df_url_posts['account_id'] = df_url_posts['account_id'].astype(int)

    df_url_posts['datetime'] = pd.to_datetime(df_url_posts['datetime'])
    df_url_posts = df_url_posts.sort_values(by=['datetime'], ascending=True)
    df_url_posts = df_url_posts.drop_duplicates(subset=['url', 'account_id'])

    df_url_posts['date'] = pd.to_datetime(df_url_posts['date'])
    df_url_posts = df_url_posts[['url', 'account_id', 'date']]

    return df_url_posts


if __name__=="__main__":

    ### Figure 1 ###

    df1 = import_data('condor_bz_2.csv', 'domains_condor_data')
    df2 = import_data('condor_bz_3-4.csv', 'domains_condor_data')
    df3 = import_data('condor_bz_5+.csv', 'domains_condor_data')

    df_condor = pd.concat([df1, df2, df3])
    df_condor = clean_buzzsumo_data(df_condor)
    df_condor = df_condor[df_condor['date']!=datetime.strptime('2021-03-01', '%Y-%m-%d')]

    df_url_condor = import_data('tpfc-recent-clean.csv', 'domains_condor_data')
    df_url_condor['date'] = pd.to_datetime(df_url_condor['tpfc_first_fact_check'])

    sumup_df_1 = calculate_engagement_percentage_change_for_domains(df=df_condor, df_url=df_url_condor)
    plot_engagement_percentage_change(sumup_df_1, figure_name='figure_1')
    print_statistics(sumup_df_1)
    export_data(sumup_df_1[['domain_name', 'repeat_vs_free_percentage_change']], 'list_domains_condor', 'domains_condor_data')
    condor_domains = list(sumup_df_1['domain_name'].unique())

    ### Figure 2 ###

    df_sf_0 = import_data('sf_bz_2+.csv', 'domains_sciencefeedback_data')
    df_sf_0 = clean_buzzsumo_data(df_sf_0)
    df_sf = df_sf_0[~df_sf_0['domain_name'].isin(condor_domains)]

    df_url_sf = import_data('appearances_2021-10-21.csv', 'domains_sciencefeedback_data')
    df_url_sf['date'] = pd.to_datetime(df_url_sf['Date of publication'])

    sumup_df_2 = calculate_engagement_percentage_change_for_domains(df=df_sf, df_url=df_url_sf)
    plot_engagement_percentage_change(sumup_df_2, figure_name='figure_2')
    print_statistics(sumup_df_2)
    export_data(sumup_df_2[['domain_name', 'repeat_vs_free_percentage_change']], 'list_domains_sciencefeedback', 'domains_sciencefeedback_data')

    ### Figure 3 ###

    df_dates = import_data('manually_filtered_reduced_posts.csv', folder='groups_crowdtangle_search')
    df_dates['post_date'] = pd.to_datetime(df_dates['post_date'])
    plot_reduced_posts_dates(df_dates, figure_name='figure_3')

    ### Figure 4 ###

    df_dates = df_dates.sort_values(by=['group_url', 'post_date']).drop_duplicates(subset=['group_url'])
    df_dates['group_id'] = df_dates['group_url'].apply(lambda x: x.split('/')[-1]).astype(int)

    df_posts_1 = import_data('posts_reduced_groups_2022-01-03.csv', folder='groups_crowdtangle_search')
    df_posts_1 = clean_crowdtangle_data(df_posts_1)

    sumup_df_3 = calculate_engagement_percentage_change_for_groups(
        df_posts=df_posts_1, how='30_days_before_vs_after', df_dates=df_dates
    )
    plot_engagement_percentage_change(sumup_df_3, figure_name='figure_4')
    print_statistics(sumup_df_3)
    export_data(sumup_df_3[['group_url', 'after_vs_before_percentage_change']], 'list_groups_crowdtangle_search', 'groups_crowdtangle_search')

    ### Figure 5 ###

    df_posts_2 = import_data('posts_repeat_offender_groups_2021-12-15.csv', folder='groups_sciencefeedback_data')
    df_posts_2 = df_posts_2[~df_posts_2['account_id'].isin(df_posts_1['account_id'].unique())]
    df_posts_2 = clean_crowdtangle_data(df_posts_2)
    # df_posts_2 = df_posts_2[df_posts_2['date']!=datetime.strptime('2021-12-15', '%Y-%m-%d')]
    # df_posts_2 = df_posts_2[df_posts_2['date']!=datetime.strptime('2021-12-14', '%Y-%m-%d')]

    df_url = import_data('appearances_2021-12-15.csv', folder='groups_sciencefeedback_data')
    df_url['date'] = pd.to_datetime(df_url['date'])

    df_url_posts = import_data('posts_url_2021-12-15.csv', folder='groups_sciencefeedback_data')
    df_url_posts = clean_df_url_posts(df_url_posts)

    sumup_df_4 = calculate_engagement_percentage_change_for_groups(
        df_posts=df_posts_2, how='repeat_offender_vs_free_periods', df_url_posts=df_url_posts, df_url=df_url
    )
    plot_engagement_percentage_change(sumup_df_4, figure_name='figure_5')
    print_statistics(sumup_df_4)
    export_data(sumup_df_4[['group_url', 'repeat_vs_free_percentage_change']], 'list_groups_sciencefeedback', 'groups_sciencefeedback_data')
