from utils import import_data, export_data


def filter_domain_list(df):
    df = df[df['total_pages'] > 1]
    df = df[df['total_pages'] < 1000]
    return df


if __name__=="__main__":

    df = import_data('summary_sf_bz_unfiltered_2+.csv', 'domains_sciencefeedback_data')
    print(len(df))
    df = filter_domain_list(df)
    print(len(df))
    export_data(df, 'summary_sf_bz_2+', 'domains_sciencefeedback_data')
