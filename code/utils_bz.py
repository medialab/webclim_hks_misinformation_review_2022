from datetime import datetime, timedelta
import time
import csv
import json

import requests


data_to_keep = [
    'id',
    'url',
    'title',
    'published_date',
    'updated_at',
    'domain_name',
    'total_shares',
    'total_facebook_shares',
    'facebook_likes',
    'facebook_comments',
    'facebook_shares',
    'twitter_shares',
    'pinterest_shares',
    'total_reddit_engagements',
    'alexa_rank',
]


def call_buzzsumo_once(params):

    api_call_attempt = 0

    while True:

        if api_call_attempt == 0:
            start_call_time = time.time()
        # If second try or more, wait an exponential amount of time (first 2 sec, then 4, then 16...)
        else: 
            time.sleep(2**(api_call_attempt - 1))
        
        r = requests.get('https://api.buzzsumo.com/search/articles.json', params=params)
        #print(r.status_code)

        # If first try, add a sleep function so that we wait at least 1.05 seconds between two calls
        if api_call_attempt == 0:
            end_call_time = time.time()
            if (end_call_time - start_call_time) < 1.2:
                time.sleep(1.2 - (end_call_time - start_call_time))

        if r.status_code == 200:
            break
        else:
            print(r.status_code, r.json())
            api_call_attempt += 1
            if r.status_code == 500:
                print(json.dumps(params, indent=4))

    return r.json()


def collect_buzzsumo_data_for_one_domain_day_by_day(domain, token, output_path, begin_date, end_date):
    
    params = {
        'q': domain,
        'api_key': token,
        'num_results': 100
    }

    begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    number_of_days_to_collect = (end_date - begin_date).days

    f = open(output_path, 'w')
    with f:
        writer = csv.writer(f)
        writer.writerow(data_to_keep)

        begin_date_temp = begin_date

        for day_index in range(number_of_days_to_collect):

            end_date_temp = begin_date_temp + timedelta(days=1)

            params['begin_date'] = begin_date_temp.timestamp()
            params['end_date'] = end_date_temp.timestamp()

            r = call_buzzsumo_once(params)
            for result in r['results']:
                writer.writerow([result[column_name] for column_name in data_to_keep])
            
            begin_date_temp = end_date_temp


def get_nb_pages_per_period_dates(period_dates, params):
    
    nb_pages = []
    
    for period_dates_index in range(len(period_dates) - 1):

        params['begin_date'] = period_dates[period_dates_index]
        params['end_date'] = period_dates[period_dates_index + 1]

        r = call_buzzsumo_once(params)
        nb_pages.append(r['total_pages'])
    
    return nb_pages


def optimize_period_dates_wrt_nb_pages(period_dates, nb_pages):

    new_period_dates = period_dates

    if any(nb_page > 99 for nb_page in nb_pages):
        for nb_page_index in range(len(nb_pages)):
            if nb_pages[nb_page_index] > 99:
                new_period_dates.append((period_dates[nb_page_index] + period_dates[nb_page_index + 1]) / 2)
 
        new_period_dates.sort()

    return new_period_dates


def collect_buzzsumo_data_for_one_domain(domain, token, begin_date, end_date, writer):

    params = {
        'q': domain,
        'api_key': token,
        'num_results': 100
    }

    period_dates = [
        datetime.strptime(begin_date, '%Y-%m-%d').timestamp(), 
        datetime.strptime(end_date, '%Y-%m-%d').timestamp()
    ]

    # Here we optimize the periods used to create the API requests, because Buzzsumo
    # prevents us from getting more than 100 pages for one request, thus we create 
    # adapted time periods that return less than 100 pages.
    nb_pages = [1000]
    while any(nb_page > 99 for nb_page in nb_pages):
        nb_pages = get_nb_pages_per_period_dates(period_dates, params)
        period_dates = optimize_period_dates_wrt_nb_pages(period_dates, nb_pages)

    for period_dates_index in range(len(period_dates) - 1):

        params['begin_date'] = period_dates[period_dates_index]
        params['end_date'] = period_dates[period_dates_index + 1]
        page = 0

        while True:

            params['page'] = page
            r = call_buzzsumo_once(params)

            if r['results'] == []:
                break
            else:
                for result in r['results']:
                    writer.writerow([result[column_name] for column_name in data_to_keep])
                page += 1