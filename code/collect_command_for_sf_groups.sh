#!/bin/bash

TODAY_DATE=$(date +"%Y-%m-%d")

minet ct posts --list-ids 1639292 --start-date 2021-01-01 --end-date 2021-12-15 > \
 "./data/posts_repeat_offender_groups_${TODAY_DATE}.csv"
