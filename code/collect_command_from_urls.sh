#!/bin/bash

INPUT_FILE="./data/appearances_2021-12-15.csv"

TODAY_DATE=$(date +"%Y-%m-%d")
OUTPUT_FILE="./data/posts_url_${TODAY_DATE}.csv"

minet ct summary url_cleaned $INPUT_FILE --posts $OUTPUT_FILE \
 --sort-by total_interactions --start-date 2021-01-01 --platforms facebook