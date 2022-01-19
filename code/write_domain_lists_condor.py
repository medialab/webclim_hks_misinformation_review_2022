import os
from datetime import datetime
import csv

from dotenv import load_dotenv
from tqdm import tqdm

from utils import import_data, export_data
from utils_bz import call_buzzsumo_once


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
    'giphy.com',
    'google.com',
    'home.blog',
    'iheart.com',
    'imgur.com'
    'instagram.com',
    'lbry.tv',
    'linkedin.com',
    'medium.com',
    'newtube.app',
    'parler.com',
    'redd.it',
    'reddit.com',
    'rumble.com',
    'scribd.com',
    'tumblr.com',
    'twitter.com',
    'vimeo.com',
    'wordpress.com',
    'youtube.com',
]


def keep_only_top_domains(df_condor, upper_limit, lower_limit):

    s = df_condor['domain'].value_counts()
    if upper_limit:
        s = s[(s <= upper_limit) & (s >= lower_limit)]
    else:
        s = s[s >= lower_limit]

    list_domain = list(s.index)

    for domain in domain_to_not_collect:
        if domain in list_domain:
            list_domain.remove(domain)
    
    return list_domain


def collect_buzzsumo_summary_per_domain(list_domain, params, output_name):

    output_path = os.path.join(".", "data", output_name)
    f = open(output_path, 'w')

    with f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'total_pages', 'total_results'])

        for domain in tqdm(list_domain):
            params['q'] = domain
            r = call_buzzsumo_once(params)
            writer.writerow([domain, r['total_pages'], r['total_results']])


def filter_domain_list(input_name, output_name):

    df_summary = import_data(input_name)

    df_summary = df_summary[df_summary['total_pages'] > 1]
    df_summary = df_summary[df_summary['total_pages'] < 1000]
    len(df_summary)

    export_data(df_summary, output_name, 'domains_condor_data')


if __name__=="__main__":

    load_dotenv()
    params = {
        'api_key': os.getenv('BUZZSUMO_TOKEN'),
        'num_results': 100,
        'begin_date': datetime.strptime('2019-01-01', '%Y-%m-%d').timestamp(),
        'end_date': datetime.strptime('2021-03-01', '%Y-%m-%d').timestamp()
    }

    df_condor = import_data('tpfc-recent-clean.csv', 'domains_condor_data')

    list_domain_5 = keep_only_top_domains(df_condor, upper_limit=None, lower_limit=5)
    collect_buzzsumo_summary_per_domain(list_domain_5, params,
                                        "summary_condor_bz_unfiltered_5+.csv")
    filter_domain_list("summary_condor_bz_unfiltered_5+.csv", "summary_condor_bz_5+")

    list_domain_3_4 = keep_only_top_domains(df_condor, upper_limit=4, lower_limit=3)
    collect_buzzsumo_summary_per_domain(list_domain_3_4, params,
                                        "summary_condor_bz_unfiltered_3-4.csv")
    filter_domain_list("summary_condor_bz_unfiltered_3-4.csv", "summary_condor_bz_3-4")

    list_domain_2 = keep_only_top_domains(df_condor, upper_limit=2, lower_limit=2)
    collect_buzzsumo_summary_per_domain(list_domain_2, params,
                                        "summary_condor_bz_unfiltered_2.csv")
    filter_domain_list("summary_condor_bz_unfiltered_2.csv", "summary_condor_bz_2")

    # list_domain = keep_only_top_domains(df_condor, upper_limit=None, lower_limit=2)
    # collect_buzzsumo_summary_per_domain(list_domain, params,
    #                                     "summary_condor_bz_unfiltered.csv")
    # filter_domain_list("summary_condor_bz_unfiltered.csv", "summary_condor_bz")