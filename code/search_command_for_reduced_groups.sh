#!/bin/bash

minet ct search "Your group's distribution is reduced due to false information" \
  --platforms facebook --start-date 2016-01-01 > ./data/search_results_reduced_groups.csv
