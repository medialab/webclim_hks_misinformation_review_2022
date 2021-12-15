import pandas as pd

from utils import (import_data, export_data)


if __name__ == "__main__":

    df = import_data("posts_url_2021-12-15.csv")
    df = df[df['account_type']=='facebook_group']
    df = df.drop_duplicates(subset=['url', 'account_id'])

    s = df["account_url"].value_counts()
    df_new = pd.DataFrame(columns=["Page or Account URL", "List"])
    df_new["Page or Account URL"] = s[s >= 15].index
    df_new["List"] = 'heloise_repeat_offender_groups_2021'

    export_data(df_new, "list_of_repeat_offender_groups_2021_for_crowdtangle")