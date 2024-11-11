import time
import logging
import pandas as pd
import numpy as np
from config import DISTANCE_FILE, ADJACENCY_FILE, ELECTRICITY_PRICE_FILE, ORDERS_FILE, PARAMS, VEHICLES
from utils import load_data, calculate_rewards
from greedy_algorithm import greedy_matching
from milp_solver import solve_vehicle_order_matching_parallel
from carpool_env import CarpoolEnv
from q_learning import QLearning
from datetime import datetime, timedelta
import gc
import json
import random

def setup_logging():
    """Configures the logging settings for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler("carpool_simulation.log"),
            logging.StreamHandler()
        ]
    )

def divide_day_into_intervals(time_interval):
    """Divides a day into equal-length time intervals."""
    intervals = np.arange(0, 24, time_interval)
    return intervals

def filter_orders_for_interval(orders_df, start_time_numeric, end_time_numeric):
    """Filters orders that are within the given time interval."""
    day_start = datetime(2024, 1, 1)  # Assuming the day starts at midnight
    start_time = day_start + timedelta(hours=start_time_numeric)
    end_time = day_start + timedelta(hours=end_time_numeric)

    return orders_df[(orders_df['lpep_pickup_datetime'] >= start_time) & (orders_df['lpep_pickup_datetime'] < end_time)]

def filter_active_vehicles(vehicles, current_time, params):
    """Filters vehicles that are active in the current time interval using numeric time."""
    return [vehicle for vehicle in vehicles if current_time - params['time_interval'] <= vehicle['time'] <= current_time]

def convert_numpy_types(obj):
    """Helper function to convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def convert_to_primitives(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {key: convert_to_primitives(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_primitives(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):  # Handle Pandas Timestamp
        return obj.isoformat()
    elif isinstance(obj, datetime):  # Handle native datetime
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_vehicle_state(vehicles, filename):
    """Write the updated vehicle state to a JSON file, converting to JSON-serializable types."""
    with open(filename, 'w') as f:
        json.dump(convert_to_primitives(vehicles), f)

def load_vehicle_state(filename):
    """Load vehicle state from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def clear_memory():
    """Explicitly free memory after each interval."""
    gc.collect()

def nearest_neighbor_matching(vehicles, orders, distance_matrix):
    """Simple nearest-neighbor matching for comparison with optimized strategies."""
    matched_orders = []
    remaining_orders = orders[:]

    for vehicle in vehicles:
        min_distance = PARAMS['order_pickup_radius']
        best_order = None
        for order in remaining_orders:
            if order['passengers'] <= PARAMS['capacity']-vehicle['passengers']:
                distance = distance_matrix[vehicle['position']][order['start']]
                if distance < min_distance:
                    min_distance = distance
                    best_order = order

        if best_order:
            matched_orders.append((vehicle, best_order))
            remaining_orders.remove(best_order)

    return matched_orders

def run_scrolling_optimization(vehicles_file, orders_df, orders, D, A, T, params, charging_strategy, strategy):
    """Runs scrolling optimization over one day divided into time intervals."""
    
    time_intervals = divide_day_into_intervals(params['time_interval'])
    total_rewards = 0
    all_routes = []
    all_charging_events = []

    day_start = 0.0  # Numeric time representing midnight

    # Start tracking total time
    overall_start_time = time.time()

    for start_hour in time_intervals:
        interval_start_time = time.time()  # Start timing the interval
        
        start_time = day_start + start_hour
        end_time = day_start + start_hour + params['time_interval']

        # Load vehicle state from file for this interval
        vehicles = load_vehicle_state(vehicles_file)

        # Filter orders for the current time interval
        interval_orders_df = filter_orders_for_interval(orders_df, start_time, end_time)
        interval_orders = [
            {
                'id': index + 1,
                'start': row['PULocationID'],
                'goal': row['DOLocationID'],
                'passengers': row['passenger_count'],
                'order_time': row['lpep_pickup_datetime']
            }
            for index, row in interval_orders_df.iterrows()
        ]

        # Filter vehicles for the current time interval
        active_vehicles = filter_active_vehicles(vehicles, current_time=start_time, params=PARAMS)
        # for vehicle in active_vehicles:
        #     print("active vehicle",vehicle['id'])

        # if len(active_vehicles) == 0:
        #     # Update all vehicles' time to the end of the interval if no active vehicles
        #     for vehicle in vehicles:
        #         if vehicle['time'] < end_time:
        #             vehicle['time'] = end_time
        #     save_vehicle_state(vehicles, vehicles_file)
        #     continue

        # if len(interval_orders) == 0:
        #     # Update all vehicles' time if no orders available
        #     for vehicle in vehicles:
        #         if vehicle['time'] < end_time:
        #             vehicle['time'] = end_time
        #     save_vehicle_state(vehicles, vehicles_file)
        #     continue

        # Run the selected optimization strategy
        vehicle_to_order_combination = {vehicle['id']: [] for vehicle in vehicles}  # Default to empty orders for each vehicle

        if strategy == 'milp_parallel':
            results, interval_reward = solve_vehicle_order_matching_parallel(
                vehicles, interval_orders, D, A, T, params, charging_strategy, order_threshold=PARAMS['combination_size_threshold'], distance_threshold=PARAMS['detour_rate_threshold'], interval_end_time = end_time
            )

            # After MILP confirms the optimal matching, populate `vehicle_to_order_combination`
            for vehicle_id, order_ids in results:
                vehicle_to_order_combination[vehicle_id] = [order for order in interval_orders if order['id'] in order_ids]

        elif strategy == 'greedy':
            # In case of greedy strategy, update based on greedy matching results
            rewards, routes, charging_events, agents = greedy_matching(
                active_vehicles, interval_orders, D, A, T, params, charging_strategy
            )
            for agent in agents:
                vehicle_id = active_vehicles[agent.env.initial_position]['id']
                vehicle_to_order_combination[vehicle_id] = agent.env.active_orders
        
        elif strategy == 'nearest_neighbor':
            matched_orders = nearest_neighbor_matching(active_vehicles, interval_orders, D)
            for vehicle, order in matched_orders:
                vehicle_to_order_combination[vehicle['id']].append(order)

        interval_profit = 0
        # Process and update each vehicle based on its optimal combination
        # print("direct input", vehicles)
        for vehicle in vehicles:
            optimal_orders = vehicle_to_order_combination.get(vehicle['id'], [])
            
            # Calculate rewards, route, and other updates for the optimal order combination
            reward, route, charging_events, final_position, final_soc, current_time, passengers, active_orders = calculate_rewards(
                vehicle, optimal_orders, D, A, T, params, charging_strategy, end_time
            )

            vehicle['time'] = current_time
            vehicle['position'] = final_position
            vehicle['soc'] = final_soc
            vehicle['passengers'] = passengers
            vehicle['active_orders'] = active_orders

            if vehicle['time'] < end_time:
                vehicle['time'] = end_time
            
            interval_profit += reward

            print("From", start_time, "to", end_time, "vehicle status", vehicle, "calculated reward", reward, "allocated orders", optimal_orders, "and route:", route)

        # Save updated vehicle state to file
        save_vehicle_state(vehicles, vehicles_file)

        total_rewards += interval_profit  # Accumulate total rewards
        # print(f"Reward for interval {start_time} to {end_time}: {interval_reward}")
        print(f"Calculated Reward for interval {start_time} to {end_time}: {interval_profit}")

        # Clear memory for the next interval
        del interval_orders_df
        del interval_orders
        del active_vehicles
        clear_memory()

        # End timing the interval and print the duration
        interval_end_time = time.time()
        print(f"Interval {start_time} to {end_time} completed in {interval_end_time - interval_start_time:.2f} seconds.")
    
    # Record total elapsed time
    overall_end_time = time.time()
    total_time_consumed = overall_end_time - overall_start_time

    # Print the total rewards and time consumed after all intervals
    print(f"Total reward across all time intervals: {total_rewards}")
    print(f"Total time consumed for all intervals: {total_time_consumed:.2f} seconds")

def main():
    # Load static data
    D, A, electricity_price, orders_df, orders = load_data()
    # vehicles_file = 'vehicles_state.json'
    # save_vehicle_state(VEHICLES, vehicles_file)
    PARAMS['electricity_price'] = electricity_price

    factors = [0.5, 0.75, 1, 1.25, 1.5]
    # factors = [1]
    original_params = PARAMS.copy()

    parameters_to_test = [
        'order_pickup_radius', 'detour_rate_threshold', 
        'vehicle_speed', 'capacity']
    # parameters_to_test =['order_pickup_radius']

    # Loop through each parameter and apply each sensitivity factor
    for param in parameters_to_test:
        print(f"\n=== Sensitivity analysis for {param} ===")
        for factor in factors:

            vehicles_file = 'vehicles_state.json'
            save_vehicle_state(VEHICLES, vehicles_file)            
            PARAMS[param] = original_params[param] * factor

            # Adjust speed-based travel time if vehicle_speed is changed
            speed = PARAMS['vehicle_speed']
            T = D / speed if param == 'vehicle_speed' else D / original_params['vehicle_speed']

            print(f"\nTesting with {param}: {PARAMS[param]}")
            strategy = 'milp_parallel'
            charging_strategy = 'optimized'

            print("Combination Filter Parameters:", "order_pickup_radius:", PARAMS['order_pickup_radius'], "combination_size_threshold:",  PARAMS['combination_size_threshold'], "detour_rate_threshold:", PARAMS['detour_rate_threshold'])
            print("Vehicle Parameters:", "vehicle_speed:", PARAMS['vehicle_speed'], "capacity:", PARAMS['capacity'], "cost_per_distance:", PARAMS['cost_per_distance'], "soc_levels:", PARAMS['soc_levels'])
            print("Passenger Parameters:", "price_per_distance:", PARAMS['price_per_distance'], "order_delay_threshold:", PARAMS['order_delay_threshold'], "unit_waiting_cost:", PARAMS['unit_waiting_cost'])

            # Run the optimization
            run_scrolling_optimization(vehicles_file, orders_df, orders, D, A, T, PARAMS, charging_strategy, strategy)

            # Reset parameters to original values after each test
            PARAMS.update(original_params)
            clear_memory()  # Clear memory after each iteration

if __name__ == "__main__":
    main()
