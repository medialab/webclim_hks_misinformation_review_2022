## Data collection

First configure a `.minetrc` file with a BuzzSumo and a CrowdTangle API token to be able to collect the data using the Python library minet.

#### Misinformation websites (Condor data)

```
python code/clean_condor_data.py
python code/write_domain_lists_condor.py

python code/collect_bz_domain.py summary_condor_bz_5+.csv condor_bz_5+.csv domains_condor_data
python code/collect_bz_domain.py summary_condor_bz_3-4.csv condor_bz_3-4.csv domains_condor_data
python code/collect_bz_domain.py summary_condor_bz_2.csv condor_bz_2.csv domains_condor_data
```

# Misinformation websites (Science Feedback data)

```
python code/clean_sciencefeedback_data_for_domains.py 2021-10-21

minet bz domain-summary domain_name data/domains_sciencefeedback_data/sf_bz_unfiltered_2+.csv --begin-date 2019-05-01 --end-date 2021-10-15 > data/domains_sciencefeedback_data/summary_sf_bz_unfiltered_2+.csv

python code/write_domain_lists_sciencefeedback.py

minet bz domain domain_name data/domains_sciencefeedback_data/summary_sf_bz_2+.csv --select domain_name  --begin-date 2019-05-01 --end-date 2021-10-15 > data/domains_sciencefeedback_data/sf_bz_2+.csv
```

#### Misinformation Facebook groups (CrowdTangle search)

To collect all the Facebook posts with the terms "Your group's distribution is reduced due to false information" from CrowdTangle, you should run (it took less than one minute):
```
./code/search_command_for_reduced_groups.sh
```

The posts collected were then manually filtered using the criteria described in the article. To collect all the posts from CrowdTangle published by the remaining groups, you should run:
```
./code/collect_command_for_reduced_groups.sh
```
The command took 11 hours to run and you should change the list id if you want to replicate our method with your CrowdTangle account.

#### Misinformation Facebook groups (Science Feedback data)

```
python code/clean_sciencefeedback_data_for_groups.py 2021-12-15
./code/collect_command_from_urls.sh # Run in 2h52
python code/filter_repeat_offender_groups.py
./code/collect_command_for_sf_groups.sh # Run in 8h
```

Again you should change the list id for the last comman if you want to replicate our method with your CrowdTangle account.

## Figures

To generate the figures from the data csv files, run:
```
python code/create_figures.py
```