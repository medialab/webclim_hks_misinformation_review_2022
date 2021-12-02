#!/bin/bash

TODAY_DATE=$(date +"%Y-%m-%d")

minet ct posts --list-ids 1632919 --start-date 2020-01-01 > \
 "./data/posts_reduced_groups_${TODAY_DATE}.csv"

# minet ct posts --list-ids 1632919 --start-date 2020-01-01 --resume \
#   -o "./data/posts_reduced_groups_${TODAY_DATE}.csv"