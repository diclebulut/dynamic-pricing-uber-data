# dynamic-pricing-uber-data

## Introduction

In this repository, I take pricing concepts (static & dynamic) and apply them to publicly available Uber Rides data, which can be downloaded here: https://www.kaggle.com/datasets/shuhengmo/uber-nyc-forhire-vehicles-trip-data-2021

Only January 2021 data, weather and taxi zone data was used to develop this pipeline, however, it could be applied to a larger dataset with input concatenation. 

Walkthroughs of the models in use can be found under 'walkthroughs' folder. 
Please visit the CLI guide and quick reference guide for running instructions. 
Please visit config.py file to make changes to main configuration components. 

## Project Brief and Outcomes
<img width="2040" height="1128" alt="image" src="https://github.com/user-attachments/assets/8a538dbb-1c0c-44a8-9d56-afc4dace451a" />
<img width="2032" height="1138" alt="image" src="https://github.com/user-attachments/assets/f85a5d77-fef1-44c3-8ade-144da992a003" />
<img width="2036" height="1128" alt="image" src="https://github.com/user-attachments/assets/e7f6e16e-f424-4b65-b3b3-efdb38e9002a" />
<img width="2034" height="1152" alt="image" src="https://github.com/user-attachments/assets/2bf05981-5ffa-4bb4-9456-91bc2f129748" />

## Setup

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
