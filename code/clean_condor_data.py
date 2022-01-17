from datetime import datetime

import pandas as pd
import ural

from utils import import_data, export_data


def clean_condor_data(df_condor):

    df_condor = df_condor.dropna(subset=['clean_url'])
    df_condor['uralized_url'] = df_condor['clean_url'].apply(lambda x: ural.normalize_url(x))
    df_condor = df_condor.drop_duplicates(subset=['uralized_url'])

    df_condor = df_condor[df_condor['tpfc_rating']=='fact checked as false']
    
    df_condor = df_condor[df_condor['public_shares_top_country'].isin(['CA', 'AU', 'GB', 'US'])]

    df_condor['date'] = pd.to_datetime(df_condor['tpfc_first_fact_check'])
    df_condor = df_condor[df_condor['date'] >= datetime.strptime('2019-01-01', '%Y-%m-%d')]

    df_condor['domain'] = df_condor['clean_url'].apply(lambda x: ural.get_domain_name(x))

    return df_condor


if __name__=="__main__":

    df_condor = import_data('tpfc-recent.csv', 'domains_condor_data')
    df_condor = clean_condor_data(df_condor)
    export_data(df_condor, 'tpfc-recent-clean', 'domains_condor_data')