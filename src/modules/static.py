"""
Static Pricing Strategy Module

Implements a baseline static pricing strategy calibrated on training data
with probabilistic acceptance rates based on price deviations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def calculate_acceptance_rate(predicted_price: float, actual_fare: float) -> float:
    """
    Calculate acceptance probability based on price difference.
    Uses sigmoid function with configurable threshold.
    
    Args:
        predicted_price: Price quoted to customer
        actual_fare: True/base fare for the ride
    
    Returns:
        float: Acceptance probability [0, 1]
    """
    if actual_fare <= 0:
        return 0.0
    
    price_diff = abs(predicted_price - actual_fare)
    max_acceptable_diff = actual_fare * 0.5  #50% deviation threshold
    
    #sigmoid-based acceptance curve
    acceptance_rate = 1.0 / (1.0 + np.exp((price_diff - max_acceptable_diff) / (max_acceptable_diff * 0.1)))
    return np.clip(acceptance_rate, 0.0, 1.0)


def train_static_model(train_data: pd.DataFrame) -> float:
    """
    Calibrate static pricing model on training data.
    
    Args:
        train_data: Training dataset with 'final_fare' and 'trip_miles' columns
    
    Returns:
        float: Calibrated price per mile
    """
    average_price_per_mile = train_data['final_fare'].sum() / train_data['trip_miles'].sum()
    return average_price_per_mile


def apply_static_pricing(
    test_data: pd.DataFrame,
    price_per_mile: float,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply static pricing strategy to test data.
    
    Args:
        test_data: Test dataset
        price_per_mile: Calibrated price per mile
        verbose: Print detailed output
    
    Returns:
        pd.DataFrame: Test data with pricing and acceptance results
    """
    results = test_data.copy()
    
    results['static_price'] = (
        results['trip_miles'] * price_per_mile 
        + results.get("tolls", 0)
        + results.get('bcf', 0)
        + results.get('sales_tax', 0)
        + results.get('congestion_surcharge', 0)
        + results.get("airport_fee", 0)
    )
    
    results['acceptance_rate'] = results.apply(
        lambda row: calculate_acceptance_rate(row['static_price'], row['final_fare']),
        axis=1
    )
    
    results['is_accepted'] = results['acceptance_rate'].apply(
        lambda rate: np.random.rand() < rate
    )
  
    results['static_profit'] = results.apply(
        lambda row: row['static_price'] - row.get('driver_pay', 0) - row.get('sales_tax', 0) 
                    if row['is_accepted'] else 0,
        axis=1
    )
    
    if verbose:
        print("\n" + "="*70)
        print("STATIC PRICING PERFORMANCE METRICS")
        print("="*70)
        print(f"Average Price Per Mile: ${price_per_mile:.2f}")
        print(f"Average Profit Per Ride: ${results['static_profit'].mean():.2f}")
        print(f"Average Fare Per Ride: ${results['static_price'].mean():.2f}")
        
        total_revenue = results[results['is_accepted']]['static_price'].sum()
        avg_revenue = total_revenue / len(results)
        
        print(f"Average Revenue Per Ride: ${avg_revenue:.2f}")
        print(f"Acceptance Rate: {results['is_accepted'].mean():.2%}")
        print(f"Accepted Rides: {results['is_accepted'].sum()} / {len(results)}")
        print(f"Total Revenue: ${total_revenue:.2f}")
    
    return results


def get_static_metrics(results: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract key metrics from static pricing results.
    
    Args:
        results: Output from apply_static_pricing()
    
    Returns:
        dict: Dictionary of performance metrics
    """
    total_revenue = results[results['is_accepted']]['static_price'].sum()
    
    return {
        'avg_price': results['static_price'].mean(),
        'avg_profit': results['static_profit'].mean(),
        'acceptance_rate': results['is_accepted'].mean(),
        'accepted_rides': results['is_accepted'].sum(),
        'total_rides': len(results),
        'total_revenue': total_revenue,
        'avg_revenue_per_ride': total_revenue / len(results),
        'price_std': results['static_price'].std(),
    }
