import ast
import numpy as np
import os
import pandas as pd
import random
import scipy.stats as stats
import time

pd.options.display.max_colwidth = 900
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.0f}'.format

from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from ural import get_domain_name
from utils import import_data
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from oauth2client.service_account import ServiceAccountCredentials
import gspread

def save_figure(figure_name):

    figure_path = os.path.join(".", "figure", figure_name + '.png')
    plt.savefig(figure_path, bbox_inches='tight')

    print(
        '\n' + figure_name.upper() + '\n',
        "The '{}' figure has been saved in the '{}' folder.\n"\
            .format(figure_path.split('/')[-1], figure_path.split('/')[-2])
        )

def save_data(df, file_name, append):

    file_path = os.path.join('.', 'data', 'iffy', file_name)

    if append == 1:
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print('{} is saved.'.format(file_name))

def get_first_strike_date():

    df_dates = import_data('manually_filtered_reduced_posts.csv', folder = 'groups_crowdtangle_search')
    df_dates = df_dates.sort_values(by=['url_group', 'post_date']).drop_duplicates(subset=['url_group'])

    df_dates['account_id'] = df_dates['url_group'].str.split('groups/').str[1]
    df_dates['account_id'] = df_dates['account_id'].astype('int64')

    return df_dates

def get_domain_names (remove_platforms):

    df = import_data('posts_reduced_groups_2022-01-03.csv', folder = 'groups_crowdtangle_search')

    df = df.dropna(subset = ['caption'])

    df_plat = import_data('platforms.csv', folder = 'iffy')
    list_platforms = df_plat['domain_name'].tolist()

    df_alt_plat = import_data('alt_platforms.csv', folder = 'iffy')
    list_alt_platforms = df_plat['domain_name'].tolist()

    if remove_platforms == 1:

        list_domains_exclude = list_platforms + list_alt_platforms
        df = df[~df['caption'].isin(list_domains_exclude)]

    df['domain_name'] = df['caption']
    df['date'] = pd.to_datetime(df['date'])
    df = df.reset_index()

    return df

def get_domains_ratings (df_domains):

    df2 = import_data('data.csv', folder = 'iffy')
    df2 = df2.rename(columns={'Domain': 'domain_name', 'MBFC factual': 'MBFC_factual'})

    df3 = df_domains

    df3['MBFC_factual'] = ''
    df3.set_index('domain_name', inplace = True)
    df3.update(df2.set_index('domain_name'))
    df3 = df3.reset_index()
    df3['MBFC_factual'] = df3['MBFC_factual'].replace('','unrated')

    return df3

def get_domains_categories ():

    df1 = import_google_sheet ('domain_names_rating')
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df_plat = import_data('platforms.csv', folder = 'iffy')
    list_platforms = df_plat['domain_name'].tolist()

    df_alt_plat = import_data('alt_platforms.csv', folder = 'iffy')
    list_alt_platforms = df_alt_plat['domain_name'].tolist()

    col_list = ['actual_like_count',
               'actual_share_count',
               'actual_comment_count',
               'actual_love_count',
               'actual_wow_count',
               'actual_haha_count',
               'actual_sad_count',
               'actual_angry_count',
               'actual_thankful_count',
               'actual_care_count']

    df2 = get_domain_names (remove_platforms = 0)
    df2['category']=''

    df2.set_index('domain_name', inplace=True)
    df2.update(df1.set_index('domain_name'))
    df2=df2.reset_index()

    df2['category'] = df2['category'].replace('','uncategorized')
    df2['engagement'] = df2[col_list].sum(axis=1)

    return df2, list_platforms, list_alt_platforms

def get_percentage_change_rating (df_domains, df_dates, days):

    df =  get_domains_ratings (df_domains)

    rating_iffy = 'MBFC_factual'

    df_percentage_rating = pd.DataFrame(columns=['account_id',
                                                 'account_name',
                                                'total_nb_links',
                                                'total_rated_negative_iffy',
                                                'nb_neg_iffy_before',
                                                'nb_neg_iffy_after',
                                                'variation_rate_nb_negative_iffy',
                                                'total_nb_links_before',
                                                'total_nb_links_after',
                                                'nb_links_before_mean_30_days',
                                                'nb_links_after_mean_30_days',
                                                'share_iffy_links_before',
                                                'share_iffy_links_after',
                                                'percentage_point_change_share_iffy_links',
                                                'account_subscriber_count'])


    remove = ['unrated', '(satire)']
    negative = ['low', 'very-low']

    print('total account_names with rated domains', len(df['account_name'].unique()))

    account_ids = [x for x in df['account_id'].unique() if x in df_dates['account_id'].unique()]

    for user in account_ids:

        df_user = df[df['account_id'] == user]

        account_subscriber_count = df_user['account_subscriber_count'].iloc[0]
        account_name = df_user['account_name'].iloc[0]

        total_nb_links = len(df_user)

        total_rated_negative_iffy = df_user[df_user[rating_iffy].isin(negative)][rating_iffy].count()

        reduced_distribution_date = df_dates[df_dates['account_id'] == user]['post_date'].values[0]
        reduced_distribution_date = datetime.strptime(str(reduced_distribution_date)[:10], '%Y-%m-%d')

        before_df = df_user[
        (df_user['date'] >= reduced_distribution_date - timedelta(days=days)) &
        (df_user['date'] < reduced_distribution_date)
        ]

        after_df = df_user[
        (df_user['date'] > reduced_distribution_date) &
        (df_user['date'] <= reduced_distribution_date + timedelta(days=days))
        ]

        total_nb_links_before = len(before_df)
        total_nb_links_after = len(after_df)

        nb_links_before_mean_30_days = round(len(before_df)/30)
        nb_links_after_mean_30_days = round(len(after_df)/30)

        total_with_rating_before_iffy = before_df[rating_iffy].count()
        total_with_rating_after_iffy = after_df[rating_iffy].count()

        nb_neg_iffy_before = before_df[before_df[rating_iffy].isin(negative)][rating_iffy].count()
        nb_neg_iffy_after = after_df[after_df[rating_iffy].isin(negative)][rating_iffy].count()

        if nb_neg_iffy_before > 0 :

            d = (100*((nb_neg_iffy_after - nb_neg_iffy_before)/ nb_neg_iffy_before))
            change_share_negative_iffy = round(d,2)

        else:
            change_share_negative_iffy = 0

        if total_nb_links_before > 0 :
            share_iffy_links_before = (100*(nb_neg_iffy_before / total_nb_links_before))
        else :
            share_iffy_links_before = 0

        if total_nb_links_after > 0 :
            share_iffy_links_after = (100*(nb_neg_iffy_after / total_nb_links_after))
        else :
            share_iffy_links_after = 0

        if share_iffy_links_before > 0 and share_iffy_links_after > 0 :
            percentage_point_change_share_iffy_links = 100*(share_iffy_links_after - share_iffy_links_before)/(share_iffy_links_before)
        else :
            percentage_point_change_share_iffy_links = 0

        df_percentage_rating = df_percentage_rating.append({
                    'account_id': user,
                    'account_name': account_name,
                    'total_nb_links': total_nb_links,
                    'total_rated_negative_iffy': total_rated_negative_iffy,
                    'nb_neg_iffy_before': nb_neg_iffy_before ,
                    'nb_neg_iffy_after': nb_neg_iffy_after ,
                    'variation_rate_nb_negative_iffy': change_share_negative_iffy,
                    'total_nb_links_before': total_nb_links_before,
                    'total_nb_links_after': total_nb_links_after,
                    'nb_links_before_mean_30_days': nb_links_before_mean_30_days,
                    'nb_links_after_mean_30_days': nb_links_after_mean_30_days,
                    'share_iffy_links_before': share_iffy_links_before,
                    'share_iffy_links_after': share_iffy_links_after,
                    'percentage_point_change_share_iffy_links' : percentage_point_change_share_iffy_links,
                    'account_subscriber_count': account_subscriber_count}, ignore_index=True)

    timestr = time.strftime('%Y_%m_%d')
    title = 'change_percentage_rating_agg_' + timestr + '.csv'

    save_data(df_percentage_rating, title, 0)

    return df_percentage_rating

def get_percentage_change_categories (df_dates, days):

    df, list_platforms, list_alt_platforms = get_domains_categories ()

    df_percentage_cat = pd.DataFrame(columns=['account_id',
                                                'account_name',
                                                'total_nb_links',
                                                'total_nb_links_before',
                                                'total_nb_links_after',
                                                'variation_rate_nb_links',
                                                'nb_platform_links',
                                                'nb_alt_platform_links',
                                                'nb_platform_links_before',
                                                'nb_platform_links_after',
                                                'variation_rate_nb_platform_links',
                                                'nb_alt_platform_links_before',
                                                'nb_alt_platform_links_after',
                                                'variation_rate_nb_alt_platform_links',
                                                'account_subscriber_count',
                                                'total_engagement_before',
                                                'total_engagement_after',
                                                'variation_rate_engagement'])


    print('total account_names with categorized domains', len(df['account_name'].unique()))

    account_ids = [x for x in df['account_id'].unique() if x in df_dates['account_id'].unique()]

    for user in account_ids:

        df_user = df[df['account_id'] == user]
        account_subscriber_count = df_user['account_subscriber_count'].iloc[0]
        account_name = df_user['account_name'].iloc[0]

        total_nb_links = len(df_user)

        total_within_category = df_user[~df_user['category'].isin(['uncategorized'])]['category'].count()

        nb_platform_links = df_user[df_user['domain_name'].isin(list_platforms)]['domain_name'].count()
        nb_alt_platform_links = df_user[df_user['domain_name'].isin(list_alt_platforms)]['domain_name'].count()

        reduced_distribution_date = df_dates[df_dates['account_id'] == user]['post_date'].values[0]
        reduced_distribution_date = datetime.strptime(str(reduced_distribution_date)[:10], '%Y-%m-%d')

        before_df = df_user[
        (df_user['date'] >= reduced_distribution_date - timedelta(days=days)) &
        (df_user['date'] < reduced_distribution_date)
        ]

        after_df = df_user[
        (df_user['date'] > reduced_distribution_date) &
        (df_user['date'] <= reduced_distribution_date + timedelta(days=days))
        ]

        nb_platform_links_before = before_df[before_df['domain_name'].isin(list_platforms)]['domain_name'].count()
        nb_platform_links_after = after_df[after_df['domain_name'].isin(list_platforms)]['domain_name'].count()

        nb_alt_platform_links_before = before_df[before_df['domain_name'].isin(list_alt_platforms)]['domain_name'].count()
        nb_alt_platform_links_after = after_df[after_df['domain_name'].isin(list_alt_platforms)]['domain_name'].count()

        total_nb_links_before = before_df['domain_name'].count()
        total_nb_links_after = after_df['domain_name'].count()

        total_engagement_before = before_df['engagement'].sum()
        total_engagement_after = after_df['engagement'].sum()

        if nb_alt_platform_links_before > 0 :

            a = (100*((nb_alt_platform_links_after - nb_alt_platform_links_before)/ nb_alt_platform_links_before))
            variation_rate_nb_alt_platform_links = round(a,2)

        else:
            variation_rate_nb_alt_platform_links = 0

        if nb_platform_links_before > 0 :

            b = (100*((nb_platform_links_after - nb_platform_links_before)/ nb_platform_links_before))
            variation_rate_nb_platform_links = round(b,2)

        else:
            variation_rate_nb_platform_links = 0

        if total_nb_links_before > 0 :

            c = (100*((total_nb_links_after - total_nb_links_before)/ total_nb_links_before))
            variation_rate_nb_links = round(c,2)

        else:
            variation_rate_nb_links = 0

        if total_engagement_before > 0 :

            d = (100*((total_engagement_after - total_engagement_before)/ total_engagement_before))
            variation_rate_engagement = round(d,2)

        else:
            variation_rate_engagement = 0

        df_percentage_cat = df_percentage_cat.append({
                    'account_id': user,
                    'account_name': account_name,
                    'total_nb_links': total_nb_links,
                    'total_nb_links_before': total_nb_links_before,
                    'total_nb_links_after': total_nb_links_after,
                    'variation_rate_nb_links': variation_rate_nb_links,
                    'nb_platform_links': nb_platform_links,
                    'nb_alt_platform_links': nb_alt_platform_links,
                    'nb_platform_links_before': nb_platform_links_before,
                    'nb_platform_links_after': nb_platform_links_after,
                    'variation_rate_nb_platform_links': variation_rate_nb_platform_links,
                    'nb_alt_platform_links_before': nb_alt_platform_links_before,
                    'nb_alt_platform_links_after': nb_alt_platform_links_after,
                    'variation_rate_nb_alt_platform_links': variation_rate_nb_alt_platform_links,
                    'account_subscriber_count': account_subscriber_count,
                    'total_engagement_before': total_engagement_before,
                    'total_engagement_after': total_engagement_after,
                    'variation_rate_engagement': variation_rate_engagement}, ignore_index=True)

    timestr = time.strftime('%Y_%m_%d')
    title = 'change_percentage_category_' + timestr + '.csv'

    save_data(df_percentage_cat, title, 0)
    return df_percentage_cat

def get_percentage_change_posts (df_dates, days):

    df = import_data('posts_reduced_groups_2022-01-03.csv', folder = 'groups_crowdtangle_search')
    df['date'] = pd.to_datetime(df['date'])

    df_percentage_posts = pd.DataFrame(columns=['account_id',
                                                'account_name',
                                                'total_nb_posts',
                                                'total_nb_posts_before',
                                                'total_nb_posts_after',
                                                'variation_rate_nb_posts',
                                                'nb_posts_before_mean_30_days',
                                                'nb_posts_after_mean_30_days',
                                                'variation_rate_daily_nb_posts',
                                                'account_subscriber_count',
                                               ])

    account_ids = [x for x in df['account_id'].unique() if x in df_dates['account_id'].unique()]

    for user in account_ids:

        df_user = df[df['account_id'] == user]

        account_subscriber_count = df_user['account_subscriber_count'].iloc[0]
        account_name = df_user['account_name'].iloc[0]

        total_nb_posts = len(df_user)

        reduced_distribution_date = df_dates[df_dates['account_id'] == user]['post_date'].values[0]
        reduced_distribution_date = datetime.strptime(str(reduced_distribution_date)[:10], '%Y-%m-%d')

        before_df = df_user[
        (df_user['date'] >= reduced_distribution_date - timedelta(days=days)) &
        (df_user['date'] < reduced_distribution_date)
        ]

        after_df = df_user[
        (df_user['date'] > reduced_distribution_date) &
        (df_user['date'] <= reduced_distribution_date + timedelta(days=days))
        ]

        total_nb_posts_before = len(before_df)
        total_nb_posts_after = len(after_df)

        nb_posts_before_mean_30_days = (total_nb_posts_before / 30)
        nb_posts_after_mean_30_days = (total_nb_posts_after / 30)

        if total_nb_posts_before > 0 :
            a = (100*((total_nb_posts_after - total_nb_posts_before)/ total_nb_posts_before))
            variation_rate_nb_posts = round(a,2)

        else:
            variation_rate_nb_posts = 0

        if nb_posts_before_mean_30_days > 0 :
            b = (100*((nb_posts_after_mean_30_days - nb_posts_before_mean_30_days)/ nb_posts_before_mean_30_days))
            variation_rate_daily_nb_posts = round(b,2)

        else:
            variation_rate_daily_nb_posts = 0

        df_percentage_posts = df_percentage_posts.append({
                    'account_id': user,
                    'account_name': account_name,
                    'total_nb_posts': total_nb_posts ,
                    'total_nb_posts_before': total_nb_posts_before,
                    'total_nb_posts_after': total_nb_posts_after,
                    'variation_rate_nb_posts': variation_rate_nb_posts,
                    'nb_posts_before_mean_30_days': nb_posts_before_mean_30_days,
                    'nb_posts_after_mean_30_days': nb_posts_after_mean_30_days,
                    'variation_rate_daily_nb_posts': variation_rate_daily_nb_posts,
                    'account_subscriber_count': account_subscriber_count,
                    }, ignore_index=True)

    timestr = time.strftime('%Y_%m_%d')
    title = 'change_percentage_posts_' + timestr + '.csv'

    save_data(df_percentage_posts, title, 0)

    return df_percentage_posts

def get_df ():

    df_change_category = get_percentage_change_categories (df_dates = get_first_strike_date(),
                                                           days = 30)

    df_change_rating = get_percentage_change_rating (df_domains = get_domain_names (remove_platforms = 0),
                                                     df_dates = get_first_strike_date(),
                                                     days = 30)

    df_change_posts = get_percentage_change_posts (df_dates = get_first_strike_date(),
                                                   days = 30)

    return df_change_rating, df_change_category, df_change_posts

def get_median_behavioral_metrics():

    timestr = time.strftime('%Y_%m_%d')
    #timestr = '2022_03_25'
    df1 = import_data('change_percentage_rating_agg_' + timestr +'.csv', folder = 'iffy')
    df2 = import_data('change_percentage_posts_' + timestr +'.csv', folder = 'iffy')
    df3 = import_data('change_percentage_category_' + timestr +'.csv', folder = 'iffy')

    df_stat = pd.DataFrame(columns=['variable',
                                    'median 30 days before',
                                    'median 30 days after',
                                    'wilcoxon'])

    df_stat['variable'] = ['nb_iffy_links',
                           'nb_of_posts_with_alt_plat',
                           'nb_of_posts_with_plat',
                           'nb_of_posts_with_links',
                           'nb_of_posts',
                           'nb_posts_mean_30_days',
                           'nb_posts_with_links_mean_30_days',
                           'share_iffy_links_30_days_over_total']

    df_stat['median 30 days before'] = [df1['nb_neg_iffy_before'].median(),
                                        df3['nb_alt_platform_links_before'].median(),
                                        df3['nb_platform_links_before'].median(),
                                        df3['total_nb_links_before'].median(),
                                        df2['total_nb_posts_before'].median(),
                                        df2['nb_posts_before_mean_30_days'].median(),
                                        df1['nb_links_before_mean_30_days'].median(),
                                        df1['share_iffy_links_before'].median()]

    df_stat['median 30 days after'] = [df1['nb_neg_iffy_after'].median(),
                                       df3['nb_alt_platform_links_after'].median(),
                                       df3['nb_platform_links_after'].median(),
                                       df3['total_nb_links_after'].median(),
                                       df2['total_nb_posts_after'].median(),
                                       df2['nb_posts_after_mean_30_days'].median(),
                                       df1['nb_links_after_mean_30_days'].median(),
                                       df1['share_iffy_links_after'].median()]

    w1, p1 = stats.wilcoxon(df1['nb_neg_iffy_before'].tolist(),
                            df1['nb_neg_iffy_after'].tolist())

    w2, p2 = stats.wilcoxon(df3['nb_alt_platform_links_before'].tolist(),
                            df3['nb_alt_platform_links_after'].tolist())

    w3, p3 = stats.wilcoxon(df3['nb_platform_links_before'].tolist(),
                            df3['nb_platform_links_after'].tolist())

    w4, p4 = stats.wilcoxon(df3['total_nb_links_before'].tolist(),
                            df3['total_nb_links_after'].tolist())

    w5, p5 = stats.wilcoxon(df2['total_nb_posts_before'].tolist(),
                            df2['total_nb_posts_after'].tolist())
    print('Wilcoxon, p-value, nb of posts any kind over 30 days', w5, p5)

    w6, p6 = stats.wilcoxon(df2['nb_posts_before_mean_30_days'].tolist(),
                            df2['nb_posts_after_mean_30_days'].tolist())
    print('Wilcoxon, p-value, average nb of posts any kind over 30 days', w6, p6)

    w7, p7 = stats.wilcoxon(df1['nb_links_before_mean_30_days'].tolist(),
                            df1['nb_links_after_mean_30_days'].tolist())
    print('Wilcoxon, p-value, average nb of posts with links over 30 days', w7, p7)

    w8, p8 = stats.wilcoxon(df1['share_iffy_links_before'].tolist(),
                            df1['share_iffy_links_after'].tolist())
    print('Wilcoxon, p-value, share of iffy links among posts with links', w8, p8)
    df_stat['wilcoxon'] = [p1, p2, p3, p4, p5, p6, p7, p8]

    df_plot1 = df_stat[~df_stat['variable'].isin([
                           'nb_posts_mean_30_days',
                           'nb_posts_with_links_mean_30_days',
                           'share_iffy_links_30_days_over_total'])]

    df_plot2 = df_stat[df_stat['variable'].isin(['nb_posts_mean_30_days',
                                                'nb_posts_with_links_mean_30_days',
                                                'share_iffy_links_30_days_over_total'])]
    print(df_stat)
    print('median daily posts before', df2['nb_posts_before_mean_30_days'].median())
    print('median daily posts after', df2['nb_posts_after_mean_30_days'].median())

    return df_stat, df_plot1, df_plot2

def calculate_confidence_interval_median(sample):

    medians = []
    for bootstrap_index in range(1000):
        resampled_sample = random.choices(sample, k=len(sample))
        medians.append(np.median(resampled_sample))

    return np.percentile(medians, 5), np.percentile(medians, 95)

def plot_change(df, var1, var2, var3, kind, variable_name, figure_name, list_yticks, title, proportion):

    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]})
    fig.set_size_inches(8, 2.5)

    ax0.bar(
        [0.1], [np.median(df[var1])],
        color='springgreen', edgecolor='black', width=0.8, alpha=0.4
    )
    ax0.bar(
        [0.9], [np.median(df[var2])],
        color='gold', edgecolor='black', width=0.8, alpha=0.4
    )
    low_before, high_before = calculate_confidence_interval_median(df[var1].values)
    ax0.plot([0.1, 0.1], [low_before, high_before], color='black', linewidth=0.9)
    low_after, high_after = calculate_confidence_interval_median(df[var2].values)
    ax0.plot([0.9, 0.9], [low_after, high_after], color='black', linewidth=0.9)
    ax0.set_xticks([0, 1]),
    ax0.set_xticklabels(['30 days \n before', '30 days \n after'])
    ax0.set_xlabel('the notification date')
    ax0.tick_params(axis='x', which='both', length=0)
    ax0.set_xlim(-.5, 1.5)
    ax0.set_ylabel('Median {}'.format(variable_name))
    ax0.set_ylim(-.1, 20)
    ax0.set_yticks(list_yticks)
    if proportion == 1 :
        ax0.set_yticklabels(['0%', '5%', '10%', '15%'])
    ax0.set_frame_on(False)

    # Percentage change in engagement
    random_y = list(np.random.random(len(df)))
    ax1.plot(df[var3].values, random_y,
             'o', markerfacecolor='royalblue', markeredgecolor='blue', alpha=0.6,
             label='Facebook pages')

    low, high = calculate_confidence_interval_median(df[var3].values)
    ax1.plot([low, np.median(df[var3]), high],
             [0.5 for x in range(3)], '|-', color='navy',
             linewidth=2, markersize=12, markeredgewidth=2)

    ax1.set_xlabel('{} {} \n30 days after minus before the notification date'.format(kind, variable_name))
    ax1.set_xlim(-110, 135)
    ax1.set_ylim(-.1, 1.1)

    ax1.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100]),
    ax1.set_xticklabels(['-100%', '-75%', '-50%', '-25%', ' 0%', '+25%', '+50%', '+75%', '+100%'])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    if title == 1 :

        plt.title("{} misinformation Facebook groups (CrowdTangle search)".format(len(df)), loc='left')

    fig.tight_layout()

    save_figure(figure_name)

def create_figures():

    timestr = time.strftime('%Y_%m_%d')
    #timestr = '2022_03_25'
    df_change_rating = import_data('change_percentage_rating_agg_' + timestr +'.csv', folder = 'iffy')
    df_change_posts = import_data('change_percentage_posts_' + timestr +'.csv', folder = 'iffy')

    print(np.median(df_change_rating['percentage_point_change_share_iffy_links']))
    print(np.median(df_change_posts['variation_rate_daily_nb_posts']))

    plot_change(df = df_change_posts,
                var1 = 'nb_posts_before_mean_30_days',
                var2 = 'nb_posts_after_mean_30_days',
                var3 = 'variation_rate_daily_nb_posts',
                kind = 'Percentage change of',
                variable_name = 'daily posts',
                figure_name = 'posts_per_day_reduced_groups_' + timestr,
                list_yticks = [0, 5, 10, 15, 20],
                title = 1,
                proportion = 0)

    plot_change(df = df_change_rating,
                var1 = 'share_iffy_links_before',
                var2 = 'share_iffy_links_after',
                var3 = 'percentage_point_change_share_iffy_links',
                kind = 'Percentage change of the',
                variable_name = 'proportion of low/very-low \n credibility links in posts',
                figure_name = 'iffy_links_reduced_groups_' + timestr,
                list_yticks = [0, 5, 10, 15],
                title = 0,
                proportion = 1 )

def main():

    get_df ()
    get_median_behavioral_metrics()
    create_figures()

if __name__=="__main__":

    main()
