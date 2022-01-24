import sys
import os
import csv

from dotenv import load_dotenv
from tqdm import tqdm

from utils import import_data
from utils_bz import data_to_keep, collect_buzzsumo_data_for_one_domain


if __name__=="__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    folder = sys.argv[3]

    df_summary = import_data(input_file, folder)

    load_dotenv()
    token = os.getenv('BUZZSUMO_TOKEN')
    begin_date = '2019-05-01'
    end_date = '2021-10-15'

    output_path = os.path.join(".", "data", folder, output_file)
    f = open(output_path, 'w')

    with f:
        writer = csv.writer(f)
        writer.writerow(data_to_keep)

        for domain in tqdm(df_summary['domain'].to_list()):
            
            collect_buzzsumo_data_for_one_domain(domain, token, begin_date, end_date, writer)