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


def infer_strike_dates_for_domains(df_url, domain):
    strike_dates = list(df_url[df_url['domain']==domain]['date'].values)
    strike_dates.sort()
    return strike_dates


def plot_example_condor_data_website(df, df_url, website_name):

    plt.figure(figsize=(8, 2.5))
    ax = plt.subplot(111)

    df_website = df[df['domain_name']==website_name]
    plt.title("The '" + website_name + "' website")

    plt.plot(df_website.groupby(by=["date"])['total_facebook_shares'].mean(), color='royalblue')
    ax.set_ylabel("Engagement per article")

    strike_dates = infer_strike_dates_for_domains(df_url, website_name)
    for date in strike_dates:
        plt.plot([date, date], [-3000, -300], color='C3')
    plt.text(
        s='Known strikes', color='C3', fontweight='bold',
        x=np.datetime64('2020-06-01'), horizontalalignment='right', 
        y=-600, verticalalignment='top'
    )

    repeat_offender_periods = infer_repeat_offender_periods(strike_dates)
    for period in repeat_offender_periods:
        plt.axvspan(period[0], period[1], ymin=1/11, facecolor='C3', alpha=0.1)
    patch1 = mpatches.Patch(facecolor='pink', alpha=0.4, edgecolor='k')
    patch2 = mpatches.Patch(facecolor='white', alpha=0.4, edgecolor='k')
    legend = plt.legend([patch1, patch2], 
        ["Repeat offender periods\n(2 strikes in less than 90 days)", "Normal periods"],
        loc='upper left', framealpha=1
    )
    legend.get_patches()[0].set_y(6)

    plt.text(
        np.datetime64('2020-10-20'), 23000, 'Percentage change = -60%', 
        color='C3', ha='center', va='center'
    )

    plt.xlim(np.datetime64('2019-01-01'), np.datetime64('2021-02-28'))
    plt.xticks([
        np.datetime64('2019-01-01'), np.datetime64('2020-01-01'), 
        np.datetime64('2021-01-01')
    ])
    ax.tick_params(axis='x', which='both', length=0)
    plt.ylim(-3000, 30000)
    plt.locator_params(axis='y', nbins=4)
    ax.grid(axis="y")
    ax.set_frame_on(False)
    plt.tight_layout()
    save_figure('figure_5_top')


def calculate_engagement_percentage_change_for_domains(df, df_url):

    sumup_df = pd.DataFrame(columns=[
        'domain_name',
        'engagement_repeat', 
        'engagement_normal'
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
                'engagement_normal': np.mean(free_df['total_facebook_shares']),
            }, ignore_index=True)
            
    sumup_df['percentage_change_engagement'] = ((sumup_df['engagement_repeat'] - sumup_df['engagement_normal'])/
                                                sumup_df['engagement_normal']) * 100
    sumup_df = sumup_df.dropna()
    sumup_df['repeat_vs_free_percentage_change'] = sumup_df['percentage_change_engagement'].apply(lambda x: str(int(np.round(x))) + '%')

    return sumup_df


def print_statistics(sumup_df):

    print('\nSample size:', len(sumup_df))

    print('Median engagement per post normal:', np.median(sumup_df['engagement_normal']))
    print('Median engagement per post repeat:', np.median(sumup_df['engagement_repeat']))

    print('Median engagement percentage change:', np.median(sumup_df['percentage_change_engagement']))
    w, p = stats.wilcoxon(sumup_df['percentage_change_engagement'])
    print('Wilcoxon test against zero: w =', w, ', p =', p)


def plot_engagement_change(sumup_df, data):

    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]})
    fig.set_size_inches(8, 2.5)

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
    ax0.set_xticks([-0.08, 1.08], ["Normal\nperiods", "Repeat\noffender\nperiods"])
    ax0.tick_params(axis='x', which='both', length=0)
    ax0.set_xlim(-.5, 1.5)
    ax0.set_ylabel('Median engagement per article')
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
    ax1.set_ylim(-.1, 1.1)

    ax1.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xticks(
        [-100, -75, -50, -25, 0, 25, 50, 75, 100, 125], 
        ['-100%', '-75%', '-50%', '-25%', ' 0%', '+25%', '+50%', '+75%', '+100%', '+125%']
    )
    ax1.set_yticks([])
    ax1.set_frame_on(False)

    if data == 'condor':
        ax0.set_ylim(-10, 1400)
        plt.title("{} misinformation websites (Condor data)".format(len(sumup_df)), loc='left')
        fig.tight_layout()
        save_figure('figure_5_middle')
    elif data == 'sf':
        ax0.set_ylim(-2, 65)
        plt.title("{} misinformation websites (Science Feedback data)".format(len(sumup_df)), loc='left')
        fig.tight_layout()
        save_figure('figure_5_bottom')


if __name__=="__main__":

    ### Figure 5 ###

    ### top panel ###

    df_condor = import_data('condor_bz_smaller.csv', 'domains_condor_data')
    df_condor['date'] = pd.to_datetime(df_condor['date'])

    df_url_condor = import_data('tpfc-recent-clean.csv', 'domains_condor_data')
    df_url_condor['date'] = pd.to_datetime(df_url_condor['tpfc_first_fact_check'])

    plot_example_condor_data_website(df_condor, df_url_condor, website_name='100percentfedup.com')

    ### middle panel ###

    sumup_df_1 = calculate_engagement_percentage_change_for_domains(df=df_condor, df_url=df_url_condor)
    print("Engagement percentage change for the '100percentfedup.com' website:",
        sumup_df_1[sumup_df_1['domain_name']=='100percentfedup.com']['percentage_change_engagement'].values[0]
    )
    print_statistics(sumup_df_1)
    plot_engagement_change(sumup_df_1, data='condor')
    # export_data(sumup_df_1[['domain_name', 'repeat_vs_free_percentage_change']], 'list_domains_condor', 'domains_condor_data')
    condor_domains = list(sumup_df_1['domain_name'].unique())

    ### bottom panel ###

    df_sf_0 = import_data('sf_bz_smaller.csv', 'domains_sciencefeedback_data')
    df_sf = df_sf_0[~df_sf_0['domain_name'].isin(condor_domains)]
    df_sf['date'] = pd.to_datetime(df_sf['date'])

    df_url_sf = import_data('appearances_2021-10-21.csv', 'domains_sciencefeedback_data')
    df_url_sf['date'] = pd.to_datetime(df_url_sf['Date of publication'])

    sumup_df_2 = calculate_engagement_percentage_change_for_domains(df=df_sf, df_url=df_url_sf)
    print_statistics(sumup_df_2)
    plot_engagement_change(sumup_df_2, data='sf')
    # export_data(sumup_df_2[['domain_name', 'repeat_vs_free_percentage_change']], 'list_domains_sciencefeedback', 'domains_sciencefeedback_data')
