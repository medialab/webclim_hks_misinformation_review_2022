### Data collection

First configure a `.minetrc` file with a CrowdTangle token to be able to collect the data.

#### Repeat offender groups (CrowdTangle search)

To collect all the Facebook posts with the terms "Your group's distribution is reduced due to false information" from CrowdTangle, you should run (it took less than one minute):
```
./code/search_command.sh
```

To collect all the posts of reduced groups from CrowdTangle, I ran (it took 11 hours):
```
./code/collect_command.sh
```
(You should change the list id if you want to replicate our method with your CrowdTangle account.)

#### Repeat offender groups (Science Feedback data)

```
python code/clean_sciencefeedback_data_for_groups.py 2021-12-15
./code/collect_command_from_urls.sh # Run in 2h52
python code/filter_repeat_offender_groups.py
./code/collect_command_for_sf_groups.sh # Run in ~10h?
```

### Figures

To generate the figures, run:
```
python code/create_figures_for_groups.py
```