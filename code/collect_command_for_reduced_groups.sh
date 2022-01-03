#!/bin/bash

TODAY_DATE=$(date +"%Y-%m-%d")

# minet ct posts --list-ids 1632919 --start-date 2020-01-01 > \
#  "./data/posts_reduced_groups_${TODAY_DATE}.csv"

# minet ct posts --list-ids 1632919 --start-date 2020-01-01 --resume \
#   -o "./data/posts_reduced_groups_${TODAY_DATE}.csv"

# minet ct posts --list-ids 1642017 --start-date 2021-10-01 --end-date 2021-12-31 > \
#  "./data/posts_reduced_groups_${TODAY_DATE}_small.csv"

minet ct posts --list-ids 1632919 --start-date 2020-01-01 --end-date 2021-12-31 > \
 "./data/posts_reduced_groups_${TODAY_DATE}.csv"
