### Data collection

First configure a `.minetrc` file with a CrowdTangle token to be able to collect the data.

To collect all the Facebook posts with the terms "Your group's distribution is reduced due to false information" from CrowdTangle, you should run (it took less than one minute):
```
./code/search_command.sh
```

To collect all the posts of reduced groups from CrowdTangle, I ran (it took 11 hours):
```
./code/collect_command.sh
```
(You should change the list id if you want to replicate our method with your CrowdTangle account.)

### Figures

To generate the figures, run:
```
python code/create_figures.py
```