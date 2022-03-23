from datetime import datetime

import pandas as pd

from utils import import_data, save_figure, export_data


def clean_crowdtangle_data(df_posts):

    df_posts['date'] = pd.to_datetime(df_posts['date'])
    
    df_posts["share"]   = df_posts[["actual_share_count"]].astype(int)
    df_posts["comment"] = df_posts[["actual_comment_count"]].astype(int)
    df_posts["reaction"] = df_posts[["actual_like_count", "actual_favorite_count", "actual_love_count",
        "actual_wow_count", "actual_haha_count", "actual_sad_count",
        "actual_angry_count", "actual_thankful_count"]].sum(axis=1).astype(int)

    df_posts['engagement'] = df_posts[["share", "comment", "reaction"]].sum(axis=1)

    return df_posts[['account_name', 'account_id', 'account_url', 'date', 'engagement']]


def clean_buzzsumo_data(df, websites_to_exlude):

    df['date'] = [datetime.fromtimestamp(x).date() for x in df['published_date']]
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['url'])

    df = df[~df['domain_name'].isin(websites_to_exlude)]
            
    return df[['domain_name', 'date', 'total_facebook_shares', 'twitter_shares']]


if __name__=="__main__":

    websites_to_exlude = import_data('excluded_websites.csv')
    websites_to_exlude = websites_to_exlude['websites'].to_list()

    # df1 = import_data('condor_bz_2.csv', 'domains_condor_data')
    # df2 = import_data('condor_bz_3-4.csv', 'domains_condor_data')
    # df3 = import_data('condor_bz_5+.csv', 'domains_condor_data')
    # df_condor = pd.concat([df1, df2, df3])
    # df_condor = clean_buzzsumo_data(df_condor, websites_to_exlude)
    # df_condor = df_condor[df_condor['date']!=datetime.strptime('2021-03-01', '%Y-%m-%d')]
    # export_data(df_condor, 'condor_bz_smaller', 'domains_condor_data')

    df_sf = import_data('sf_bz_2+.csv', 'domains_sciencefeedback_data')
    df_sf = clean_buzzsumo_data(df_sf, websites_to_exlude)
    export_data(df_sf, 'sf_bz_smaller', 'domains_sciencefeedback_data')

    # df_posts_1 = import_data('posts_reduced_groups_2022-01-03.csv', folder='groups_crowdtangle_search')
    # df_posts_1 = clean_crowdtangle_data(df_posts_1)
    # export_data(df_posts_1, 'posts_smaller', 'groups_crowdtangle_search')

    # df_posts_2 = import_data('posts_repeat_offender_groups_2021-12-15.csv', folder='groups_sciencefeedback_data')
    # df_posts_2 = df_posts_2[~df_posts_2['account_id'].isin(df_posts_1['account_id'].unique())]
    # df_posts_2 = clean_crowdtangle_data(df_posts_2)
    # export_data(df_posts_2, 'posts_smaller', 'groups_sciencefeedback_data')
