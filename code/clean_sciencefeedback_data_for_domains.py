import sys
import datetime

import pandas as pd
import ural

from utils import (import_data, export_data)


domain_to_not_collect = [
    '2020electioncenter.com',
    'archive.org',
    'banned.video',
    'bitchute.com',
    'brandnewtube.com',
    'brighteon.com',
    'dailymotion.com',
    'd.tube',
    'facebook.com',
    'gab.com',
    'giphy.com',
    'google.com',
    'home.blog',
    'iheart.com',
    'imgur.com'
    'instagram.com',
    'lbry.tv',
    'linkedin.com',
    'me.me',
    'medium.com',
    'newtube.app',
    'parler.com',
    'redd.it',
    'reddit.com',
    'rumble.com',
    'scribd.com',
    'tiktok.com',
    'tumblr.com',
    'twitter.com',
    'vimeo.com',
    'wordpress.com',
    'youtube.com',
]


def keep_only_the_urls_considered_fake_by_facebook(url_df):

    url_df = url_df[url_df['Flag as']=='False']

    url_df = url_df[url_df['Fb flagged']=="done"]

    url_df = url_df[~(url_df['Fb correction status'].isin([
        "Corrected to Not Rated",
        "Corrected to True",
        "Corrected to Partly False"
    ]))]

    return url_df


def clean_url_format(url_df):

    url_df['url'] = url_df['url'].transform(lambda x: x.strip())

    url_df['url_cleaned'] = url_df['url']\
        .apply(lambda x: ural.normalize_url(x, 
                                            strip_protocol=False, 
                                            strip_trailing_slash=True))
    url_df['domain'] = url_df['url_cleaned'].apply(lambda x: ural.get_domain_name(x))

    # Remove the URLs that are in double in the dataframe, 
    # keeping only the first, i.e. the more recent ocurrence.
    url_df = url_df.drop_duplicates(subset = "url", keep = "first")
    url_df = url_df.drop_duplicates(subset = "url_cleaned", keep = "first")

    return url_df


def add_info_from_fact_check_table(url_df, fact_check_df):

    url_df = url_df.dropna(subset=['Item reviewed'])
    fact_check_df = fact_check_df.dropna(subset=['Items reviewed'])

    url_df = url_df.merge(fact_check_df[['Items reviewed', 'Date of publication']], 
                        left_on='Item reviewed', right_on='Items reviewed', how='left')

    return url_df


def keep_only_top_domains(df, upper_limit, lower_limit):

    s = df['domain'].value_counts()
    if upper_limit:
        s = s[(s <= upper_limit) & (s >= lower_limit)]
    else:
        s = s[s >= lower_limit]

    list_domain = list(s.index)

    for domain in domain_to_not_collect:
        if domain in list_domain:
            list_domain.remove(domain)
    
    return list_domain


if __name__ == "__main__":

    DATE = sys.argv[1]

    url_df = import_data("Appearances-Grid view " + DATE + ".csv", "domains_sciencefeedback_data")
    url_df = keep_only_the_urls_considered_fake_by_facebook(url_df)
    url_df = clean_url_format(url_df)

    fact_check_df = import_data("Reviews _ Fact-checks-Grid view " + DATE + ".csv", "domains_sciencefeedback_data")
    url_df = add_info_from_fact_check_table(url_df, fact_check_df)
    url_df = url_df[['url', 'url_cleaned', 'domain', 'Date of publication']]

    url_df['date'] = pd.to_datetime(url_df['Date of publication'])
    url_df = url_df[url_df['date'] >= datetime.datetime.strptime('2019-05-01', '%Y-%m-%d')]
    url_df = url_df[url_df['date'] <= datetime.datetime.strptime('2021-10-15', '%Y-%m-%d')]

    print("There are {} fake news urls.".format(len(url_df)))
    export_data(url_df, "appearances_" + DATE, 'domains_sciencefeedback_data')

    list_domain_2 = keep_only_top_domains(url_df, upper_limit=None, lower_limit=2)
    domain_df = pd.DataFrame(list_domain_2, columns =['domain_name'])
    export_data(domain_df, 'sf_bz_unfiltered_2+', 'domains_sciencefeedback_data')
