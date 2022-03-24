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

#googlesheet
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


def import_google_sheet (filename):

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('./credentials.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(filename)
    sheet_instance = sheet.get_worksheet(0)
    records_data = sheet_instance.get_all_records()
    records_df = pd.DataFrame.from_dict(records_data)

    return records_df


def get_first_strike_date():

    df_dates = import_data('manually_filtered_reduced_posts.csv', folder = 'groups_crowdtangle_search')
    df_dates = df_dates.sort_values(by=['url_group', 'post_date']).drop_duplicates(subset=['url_group'])

    df_dates['account_id'] = df_dates['url_group'].str.split('groups/').str[1]
    df_dates['account_id'] = df_dates['account_id'].astype('int64')

    return df_dates

def get_domain_names (remove_platforms):

    df = import_data('posts_reduced_groups_2022-01-03.csv', folder = 'groups_crowdtangle_search')
    #print('length df before removing nan links', len(df))

    df = df.dropna(subset = ['caption'])
    #print('length df after nan caption', len(df))

    # df1 = import_google_sheet ('domain_names_rating')
    # df1 = df1.replace(r'^\s*$', np.nan, regex=True)
    #
    # list_platforms = df1[df1['category'].isin(['platform'])]['domain_name'].unique().tolist()
    # list_alt_platforms = df1[df1['category'].isin(['alternative_platform'])]['domain_name'].unique().tolist()

    df_plat = import_data('platforms.csv', folder = 'iffy')
    list_platforms = df_plat['domain_name'].tolist()

    df_alt_plat = import_data('alt_platforms.csv', folder = 'iffy')
    list_alt_platforms = df_plat['domain_name'].tolist()

    if remove_platforms == 1:

        #list_domains_exclude = ['youtube.com', 'youtu.be', 'rumble.com', 'bitchute.com', 'msn.com', 'yahoo.com', 'google.com', '', 'twitter.com', 'goo.gl', 't.co']
        list_domains_exclude = list_platforms + list_alt_platforms
        df = df[~df['caption'].isin(list_domains_exclude)]
        #print('length df after removing platforms', len(df))

    df['domain_name'] = df['caption']
    df['date'] = pd.to_datetime(df['date'])
    df = df.reset_index()

    return df

def get_domains_ratings (df_domains):

    # df1 = import_google_sheet ('domain_names_rating')
    # df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df2 = import_data('data.csv', folder = 'iffy')
    df2 = df2.rename(columns={'Domain': 'domain_name', 'MBFC factual': 'MBFC_factual'})

    df3 = df_domains

    # df3['aggregated_rating'] = ''
    # df3.set_index('domain_name', inplace = True)
    # df3.update(df1.set_index('domain_name'))
    # df3 = df3.reset_index()

    df3['MBFC_factual'] = ''
    df3.set_index('domain_name', inplace = True)
    df3.update(df2.set_index('domain_name'))
    df3 = df3.reset_index()


    df3['MBFC_factual'] = df3['MBFC_factual'].replace('','unrated')
    #df3['aggregated_rating'] = df3['aggregated_rating'].replace('','unrated')

    return df3

def get_domains_categories ():

    df1 = import_google_sheet ('domain_names_rating')
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    # list_platforms = df1[df1['category'].isin(['platform'])]['domain_name'].unique().tolist()
    # list_alt_platforms = df1[df1['category'].isin(['alternative_platform'])]['domain_name'].unique().tolist()
    #
    # df_platforms = pd.DataFrame(columns=['domain_name'])
    # df_platforms['domain_name'] = list_platforms
    # save_data(df_platforms, 'platforms.csv', 0)
    #
    # df_alt_platforms = pd.DataFrame(columns=['domain_name'])
    # df_alt_platforms['domain_name'] = list_alt_platforms
    # save_data(df_alt_platforms, 'alt_platforms.csv', 0)

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

    #rating = 'aggregated_rating'
    rating_iffy = 'MBFC_factual'

    df_percentage_rating = pd.DataFrame(columns=['account_id',
                                                 'account_name',
                                                'total_nb_links',
                                                #'total_with_rating',
                                                'total_rated_negative_iffy',
                                                #'percentage_negative_iffy',
                                                'nb_neg_iffy_before',
                                                'nb_neg_iffy_after',
                                                'variation_rate_nb_negative_iffy',
                                                'total_nb_links_before',
                                                'total_nb_links_after',
                                                'nb_links_before_mean_30_days',
                                                'nb_links_after_mean_30_days',
                                                'share_iffy_links_before',
                                                'share_iffy_links_after',
                                                'account_subscriber_count'])


    remove = ['unrated', '(satire)']
    #positive = ['high', 'very-high', 'mostly-factual']
    negative = ['low', 'very-low']
    #mixed = ['mixed']

    print('total account_names with rated domains', len(df['account_name'].unique()))

    account_ids = [x for x in df['account_id'].unique() if x in df_dates['account_id'].unique()]

    for user in account_ids:

        df_user = df[df['account_id'] == user]

        account_subscriber_count = df_user['account_subscriber_count'].iloc[0]
        account_name = df_user['account_name'].iloc[0]

        total_nb_links = len(df_user)

        #df_user = df_user[~df_user[rating].isin(remove)]

        #total_with_rating = df_user[rating].count()

        total_rated_negative_iffy = df_user[df_user[rating_iffy].isin(negative)][rating_iffy].count()

        # if total_with_rating > 0:
        #
        #     percentage_negative_iffy = round((rating_negative_iffy / total_with_rating)*100, 2)
        #
        # else:
        #
        #     percentage_negative_iffy = 0

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
            share_iffy_links_before = round(100*(nb_neg_iffy_before / total_nb_links_before))
        else :
            share_iffy_links_before = 0

        if total_nb_links_after > 0 :
            share_iffy_links_after = round(100*(nb_neg_iffy_after / total_nb_links_after))
        else :
            share_iffy_links_after = 0

        df_percentage_rating = df_percentage_rating.append({
                    'account_id': user,
                    'account_name': account_name,
                    'total_nb_links': total_nb_links,
                    #'total_with_rating': total_with_rating,
                    'total_rated_negative_iffy': total_rated_negative_iffy,
                    #'percentage_negative_iffy': percentage_negative_iffy,
                    'nb_neg_iffy_before': nb_neg_iffy_before ,
                    'nb_neg_iffy_after': nb_neg_iffy_after ,
                    'variation_rate_nb_negative_iffy': change_share_negative_iffy,
                    'total_nb_links_before': total_nb_links_before,
                    'total_nb_links_after': total_nb_links_after,
                    'nb_links_before_mean_30_days': nb_links_before_mean_30_days,
                    'nb_links_after_mean_30_days': nb_links_after_mean_30_days,
                    'share_iffy_links_before': share_iffy_links_before,
                    'share_iffy_links_after': share_iffy_links_after,
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

        df_percentage_posts = df_percentage_posts.append({
                    'account_id': user,
                    'account_name': account_name,
                    'total_nb_posts': total_nb_posts ,
                    'total_nb_posts_before': total_nb_posts_before,
                    'total_nb_posts_after': total_nb_posts_after,
                    'variation_rate_nb_posts': variation_rate_nb_posts,
                    'nb_posts_before_mean_30_days': nb_posts_before_mean_30_days,
                    'nb_posts_after_mean_30_days': nb_posts_after_mean_30_days,
                    'account_subscriber_count': account_subscriber_count,
                    }, ignore_index=True)

    timestr = time.strftime('%Y_%m_%d')
    title = 'change_percentage_posts_' + timestr + '.csv'

    save_data(df_percentage_posts, title, 0)
    #print('Median variation rate of posts:', df_percentage_posts['variation_rate_nb_posts'].median())

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

def percentage_rating_template(ax, stat, median1, m1, m2):

    #plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
    plt.vlines(x= median1 , ymin=m1-0.2, ymax=m2+0.2, color='gray', linestyle='--', linewidth=1)
    plt.text(median1+0.53, m2+0.1, 'median', fontsize=7, color='blue')

    if stat == 1 :
        plt.xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', ' 100%'])
        plt.xlim(-1, 101)

    elif stat == 2:

        plt.ticklabel_format(style = 'plain')

    elif stat == 3:
        plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100], ['-100%', '-75%', '-50%', ' -25%', '0%', '25%', '50%', '75%', ' 100%'])
        plt.xlim(-101, 101)


    plt.yticks([])
    ax.set_frame_on(False)

def calculate_confidence_interval_median(sample):

    medians = []
    for bootstrap_index in range(1000):
        resampled_sample = random.choices(sample, k=len(sample))
        medians.append(np.median(resampled_sample))

    return np.percentile(medians, 5), np.percentile(medians, 95)

def plot_bubbles(df_per, rating, xlabel, title, stat):

    plt.figure(figsize=(7, 2.5))
    ax = plt.subplot(111)

    random_y1 = np.random.random(len(df_per)).tolist()

    plt.plot(df_per[rating].values,
             random_y1,
             'o',
             markerfacecolor='blue',
             markeredgecolor='blue',
             alpha=0.6)

    plt.title('{} misinformation Facebook groups'.format(len(df_per)))

    median1 = np.median(df_per[rating])
    m1 = min(random_y1)
    m2 = max(random_y1)

    percentage_rating_template(ax,
                            stat,
                            median1,
                            m1,
                            m2)

    low, high = calculate_confidence_interval_median(df_per[rating].values)

    plt.plot([low, np.median(df_per[rating]), high],
             [0.5 for x in range(3)], '|-', color='navy',
             linewidth=2, markersize=12, markeredgewidth=2)

    plt.ylim(-.2, 1.2)

    plt.xlabel(xlabel, size='large')

    plt.tight_layout()
    figure_path = title
    save_figure(figure_path)

def plot_word_cloud(cloud_name, df_domains):

    df = df_domains
    text = str(df['account_name'].unique())
    stopwords = set(STOPWORDS)
    stopwords.update(['https', 'inbox', 'hi', 'app', 'newsletter', 'bulletins'])

    wc = WordCloud(width=300,
                    height=250,
                    background_color='white',
                    max_words=500,
                    stopwords=stopwords,
                    max_font_size=90,
                    colormap='Blues',
                    collocations = False,
                    contour_width=3,
                    contour_color='blue'
                    ).generate(text)

    plt.figure(figsize= (5,10))
    plt.axis('off')
    plt.tight_layout(pad=0)

    wc.to_file('./figure/' + cloud_name + '.jpg')

def plot_figures():

    timestr = time.strftime('%Y_%m_%d')
    #timestr = '2022_03_18'

    plot_bubbles(df_per = import_data ('change_percentage_rating_agg_' + timestr +'.csv', folder = 'iffy'),
                 rating = 'variation_rate_nb_negative_iffy',
                 xlabel = 'Variation rate of posts containing domain names \n with low and very low ratings (iffy) 30 days before and after the screenshot ' ,
                 title = 'change_negative_rating_iffy_' + timestr ,
                 stat = 2 )

    plot_bubbles(df_per = import_data ('change_percentage_category_' + timestr +'.csv', folder = 'iffy'),
                 rating = 'variation_rate_nb_platform_links',
                 xlabel = 'Variation rate of posts containing domain names \n of platforms 30 days before and after the screenshot ' ,
                 title = 'change_category_platform_' + timestr ,
                 stat = 2 )

    plot_bubbles(df_per = import_data ('change_percentage_category_' + timestr +'.csv', folder = 'iffy'),
                 rating = 'variation_rate_nb_alt_platform_links',
                 xlabel = 'Variation rate of posts containing domain names \n of alternative platforms 30 days before and after the screenshot ' ,
                 title = 'change_category_alt_platform_' + timestr ,
                 stat = 2 )

    plot_bubbles(df_per = import_data ('change_percentage_category_' + timestr +'.csv', folder = 'iffy'),
             rating = 'variation_rate_nb_links',
             xlabel = 'Variation rate of posts containing a link \n  30 days before and after the screenshot ' ,
             title = 'change_nb_links_' + timestr ,
             stat = 2 )

    # plot_bubbles(df_per = import_data ('change_percentage_category_' + timestr +'.csv', folder = 'iffy'),
    #      rating = 'variation_rate_engagement',
    #      xlabel = 'Variation rate of engagement \n  30 days before and after the screenshot ' ,
    #      title = 'change_engagement_' + timestr ,
    #      stat = 2 )

    plot_bubbles(df_per = import_data ('change_percentage_posts_' + timestr +'.csv', folder = 'iffy'),
         rating = 'variation_rate_nb_posts',
         xlabel = 'Variation rate of the number of posts \n  30 days before and after the screenshot ' ,
         title = 'change_posts_' + timestr ,
         stat = 2 )

    plot_word_cloud(cloud_name = 'cloud_account_names',
                   df_domains = import_data ('change_percentage_category_' + timestr +'.csv', folder = 'iffy') )

def get_median_behavioral_metrics():

    timestr = time.strftime('%Y_%m_%d')
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
    return df_stat, df_plot1, df_plot2

def calculate_confidence_interval_median(sample):

    medians = []
    for bootstrap_index in range(1000):
        resampled_sample = random.choices(sample, k=len(sample))
        medians.append(np.median(resampled_sample))

    return np.percentile(medians, 5), np.percentile(medians, 95)


def plot_engagement_change(df1, df2):

    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    fig.set_size_inches(5, 3)

    # Median nb of posts before VS after
    ax0.bar(
        [0.1], [np.median(df2['nb_posts_before_mean_30_days'])],
        color='springgreen', edgecolor='black', width=0.8, alpha=0.4
    )
    ax0.bar(
        [0.9], [np.median(df2['nb_posts_after_mean_30_days'])],
        color='gold', edgecolor='black', width=0.8, alpha=0.4
    )
    low_before, high_before = calculate_confidence_interval_median(df2['nb_posts_before_mean_30_days'].values)
    ax0.plot([0.1, 0.1], [low_before, high_before], color='black', linewidth=0.9)
    low_after, high_after = calculate_confidence_interval_median(df2['nb_posts_after_mean_30_days'].values)
    ax0.plot([0.9, 0.9], [low_after, high_after], color='black', linewidth=0.9)
    ax0.set_xticks([0, 1]),
    ax0.set_xticklabels(['30 days \n before', '30 days \n after'])
    ax0.set_xlabel('the notification date')
    ax0.tick_params(axis='x', which='both', length=0)
    ax0.set_xlim(-.5, 1.5)
    ax0.set_ylabel('Median of posts per day')
    ax0.set_ylim(-.1, 15)
    ax0.set_yticks([0, 5, 10, 15, 20])
    ax0.set_frame_on(False)

    #    # Median engagements before or free VS after or repeat
    ax1.bar(
        [0.1], [np.median(df1['share_iffy_links_before'])],
        color='springgreen', edgecolor='black', width=0.8, alpha=0.4
    )
    ax1.bar(
        [0.9], [np.median(df1['share_iffy_links_before'])],
        color='gold', edgecolor='black', width=0.8, alpha=0.4
    )
    low_before, high_before = calculate_confidence_interval_median(df1['share_iffy_links_before'].values)
    ax1.plot([0.1, 0.1], [low_before, high_before], color='black', linewidth=0.9)
    low_after, high_after = calculate_confidence_interval_median(df1['share_iffy_links_before'].values)
    ax1.plot([0.9, 0.9], [low_after, high_after], color='black', linewidth=0.9)
    #ax1.set_xticks([0, 1], ['30 days \n before', '30 days \n after'])
    ax1.set_xticks([0, 1]),
    ax1.set_xticklabels(['30 days \n before', '30 days \n after'])
    #ax1.set_xticklabels(
    #labels = ['30 days\nbefore', '30 days\nafter'])
    ax1.set_xlabel('the notification date')
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.set_xlim(-.5, 1.5)
    ax1.set_ylabel('Median share of links \n rated low or very-low')
    ax1.set_ylim(-.1, 20)
    ax1.set_frame_on(False)


    # Percentage change in engagement

    fig.suptitle("{} Facebook groups \n self-declared as being under reduced distribution".format(len(df1)))
    #fig.suptitle
    fig.tight_layout()
    save_figure('figure_post_links')

def bar_template(ax):

    ax.set_frame_on(False)
    ax.set(xlabel=None)
    ax.axes.get_yaxis().set_visible(False)

    for p in ax.patches:
        a = round(p.get_height())
        ax.annotate(a, (p.get_x()+p.get_width()/2. - 0.04, p.get_height()))

        if a == 10:
            a = str(a) + '%'
            ax.annotate(a, (p.get_x()+p.get_width()/2. - 0.04, p.get_height()))

def plot_bars(iffy, df):

    if iffy == 1:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 3)
        pvalues = df['wilcoxon'].tolist()

        df.plot.bar(x = 'variable', y = ['median 30 days before', 'median 30 days after'], rot = 360, ax = ax, color = ['springgreen', 'gold'], width = 0.8, edgecolor='black', alpha=0.4)
        ax.set_xticklabels(
        labels = ['Nb of posts with \n Iffy links \n \n (p-value={})'.format(round(pvalues[0],3)) ,
        'Nb of posts with \n alt. platform links \n \n(p-value={})'.format(round(pvalues[1],3)),
        'Nb of posts with \n platform links\n \n(p-value={})'.format(round(pvalues[2],3)),
        'Nb of posts \n with links\n \n (p-value={})'.format(round(pvalues[3],3)),
        'Nb of posts \n all kind \n \n (p-value={})'.format(round(pvalues[4],3))])

        #ax.set_yticks([0, 5, 10, 15, 20])
        #ax.set_yticks()
        ax.set_ylim(-.1, 500)
        bar_template(ax)

        save_figure('iffy_links_wilcoxon')

    elif iffy == 0:

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 3)
        pvalues = df['wilcoxon'].tolist()

        df.plot.bar(x = 'variable', y = ['median 30 days before', 'median 30 days after'], rot = 360, ax = ax, color = ['springgreen', 'gold'], width=0.4, edgecolor='black', alpha=0.4)
        # ax.set_xticklabels(
        # labels = [
        # 'Number of posts per day \n any kind \n \n (p-value={})'.format(round(pvalues[0],3)),
        # 'Number of posts per day \n containing a link \n \n (p-value={})'.format(round(pvalues[1],3)),
        # 'Share of "Iffy links" \n among posts containing links \n \n (p-value={})'.format(round(pvalues[2],3))])

        ax.set_xticklabels(
        labels = [
        'Number of posts per day',
        'Number of posts per day \n containing a link',
        'Share of "Iffy links" \n among posts containing links'])

        #ax.set_yticks()
        ax.set_ylim(-.1, 20)
        bar_template(ax)
        ax.legend(labels = ['Median, 30-day period before the notification date', 'Median, 30-day period after the notification date'])
        save_figure('posts_wilcoxon')

def main():

    #df_change_rating, df_change_category, df_change_posts = get_df()
    #df_stat, df_plot1, df_plot2 = get_median_behavioral_metrics()
    timestr = time.strftime('%Y_%m_%d')
    df1 = import_data('change_percentage_rating_agg_' + timestr +'.csv', folder = 'iffy')
    df2 = import_data('change_percentage_posts_' + timestr +'.csv', folder = 'iffy')
    plot_engagement_change(df1, df2)
    #plot_figures()
    #plot_bars(iffy = 1, df = df_plot1 )
    #plot_bars(iffy = 0, df = df_plot2 )


if __name__=="__main__":

    main()
