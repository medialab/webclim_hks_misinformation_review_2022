#!/bin/bash

TODAY_DATE=$(date +"%Y-%m-%d")

minet ct posts --list-ids 1632919 --start-date 2020-01-01 --end-date 2021-12-31 > \
 "./data/posts_reduced_groups_${TODAY_DATE}.csv"
