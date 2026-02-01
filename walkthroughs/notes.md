new dataset because 
- more features
- cost information
- information on driver side
- more data points to make models more robust

Info on new dataset
- Only January 2021 is used due to dataset size and processing constraints

Multicollinearity
- trip time & trip miles (pick one)
- tip & tipper flag (tip info should be removed)
- final fare & uber profit & driver final pay

Fare measure
- final fare

Uber profit measures (try separately with all 3)
- uber profit
- uber_profit_per_second
- uber_profit_per_mile

Steps taken to pick features
- Variance Inflation Factor (VIF) analysis
- Basic correlation matrix

Model Plans
passenger_model = uber_trips[['PULocationID', 'DOLocationID', 'trip_miles',
                              'tips','shared_request_flag', 'shared_match_flag',
                              'final_fare', 'wait', 'day_of_month',
                            'time_of_day', 'precip', 'preciptype']]

driver_model = uber_trips[['PULocationID', 'DOLocationID', 'trip_miles','out_of_base_dispatch_flag',
                           'driver_final_pay', 'wait', 'day_of_month',
                            'time_of_day', 'precip', 'preciptype']]

uber_side_passenger_model = [['PULocationID', 'DOLocationID', 'trip_miles',
                              'tips','shared_request_flag', 'shared_match_flag',
                              'uber_profit', 'wait', 'day_of_month',
                            'time_of_day', 'precip', 'preciptype']]