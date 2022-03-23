from datetime import datetime, timedelta
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

from utils import import_data, save_figure, export_data
from create_old_figures import (
    plot_engagement_percentage_change, 
    print_statistics,
    clean_df_url_posts,
    infer_strike_dates_for_groups,
    infer_repeat_offender_periods,
    keep_free_data,
    keep_repeat_offender_data
)

pd.options.mode.chained_assignment = None
pd.set_option('use_inf_as_na', True)


def calculate_engagement_percentage_change_for_groups(df_posts, df_dates):

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

        reduced_distribution_date = df_dates[df_dates['group_id_randomized']==group_id]['post_date'].values[0]
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

    return sumup_df


if __name__=="__main__":

    df_dates = import_data('manually_filtered_reduced_posts.csv', folder='groups_crowdtangle_search')
    df_dates['post_date'] = pd.to_datetime(df_dates['post_date'])
    df_dates = df_dates.sort_values(by=['group_url', 'post_date']).drop_duplicates(subset=['group_url'])
    df_dates['group_id'] = df_dates['group_url'].apply(lambda x: x.split('/')[-1]).astype(int)

    df_dates = df_dates.sample(frac=1)
    df_dates['group_id_randomized'] = df_dates['group_id'].shift(-1)
    df_dates['group_id_randomized'].iloc[-1] = df_dates['group_id'].iloc[0]

    df_posts_1 = import_data('posts_smaller.csv', folder='groups_crowdtangle_search')
    df_posts_1['date'] = pd.to_datetime(df_posts_1['date'])

    sumup_df_1 = calculate_engagement_percentage_change_for_groups(
        df_posts=df_posts_1, df_dates=df_dates
    )
    plot_engagement_percentage_change(sumup_df_1, figure_name='figure_4_bis')
    print_statistics(sumup_df_1)
