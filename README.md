# dynamic-pricing

In this repository, I take pricing concepts (static & dynamic) and apply them to publicly available Uber Rides data, which can be downloaded here: https://www.kaggle.com/datasets/shuhengmo/uber-nyc-forhire-vehicles-trip-data-2021

Only January 2021 data, weather and taxi zone data was used to develop this pipeline, however, it could be applied to a larger dataset with input concatenation. 

Walkthroughs of the models in use can be found under 'walkthroughs' folder. 
Please visit the CLI guide and quick reference guide for running instructions. 
Please visit config.py file to make changes to main configuration components. 

Setup:
- Clone the repository.
- Download "nyc 2021-01-01 to 2021-12-31.csv" (or any month you'd like) from https://www.kaggle.com/datasets/shuhengmo/uber-nyc-forhire-vehicles-trip-data-2021 and place it in the src/data folder.
- In terminal, run (in this order)
```
poetry install
```
```
cd src
```
```
poetry run python pipeline.py --save-markdown --markdown-complexity summary #test
```
- If poetry is not installed, install poetry first and try the previous step again.
