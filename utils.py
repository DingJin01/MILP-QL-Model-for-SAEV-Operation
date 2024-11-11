# utils.py

import itertools
import copy
import pandas as pd
from carpool_env import CarpoolEnv
from q_learning import QLearning

def filter_orders_for_vehicle(vehicle, orders, D, order_pickup_radius):
    """Filters orders based on the vehicle's pickup radius."""
    # Check that vehicle is a dictionary and has the correct keys
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle should be a dictionary.")
    if 'position' not in vehicle:
        raise KeyError("'position' key is missing in vehicle dictionary.")

    # Ensure orders is a list of dictionaries
    if not isinstance(orders, list) or not all(isinstance(order, dict) for order in orders):
        raise TypeError("orders should be a list of dictionaries.")
    
    # print(vehicle['id'],vehicle['position'])

    return [order for order in orders if D[order['start'], vehicle['position']] <= order_pickup_radius]

def load_data():
    """Loads all necessary data from CSV files."""
    D = pd.read_csv('distance.csv', header=None).values
    A = pd.read_csv('adjacency.csv', header=None).values
    electricity_price = pd.read_csv('electricity_price.csv', header=None).values
    
    # Load orders data and parse 'lpep_pickup_datetime' as datetime
    orders_df = pd.read_csv('orders.csv', parse_dates=['lpep_pickup_datetime'])
    
    # Convert orders to list of dictionaries for other uses
    orders = [
        {
            'id': index + 1,
            'start': row['PULocationID'],
            'goal': row['DOLocationID'],
            'passengers': row['passenger_count'],
            'order_time': row['lpep_pickup_datetime']  # Parsed as datetime
        }
        for index, row in orders_df.iterrows()
    ]
    
    # Return both the DataFrame and the list of orders
    return D, A, electricity_price, orders_df, orders

def create_combinations(feasible_orders, max_combination_size=None):
    """Generates all possible combinations of feasible orders up to a maximum size."""
    if max_combination_size is None:
        max_combination_size = len(feasible_orders)
    for r in range(1, max_combination_size + 1):
        for combination in itertools.combinations(feasible_orders, r):
            yield combination

def calculate_rewards(vehicle, orders, D, A, T, params, charging_strategy, end_time, seed=42):
    """
    Calculates the total rewards, route, and charging events for a given vehicle and set of orders.

    Args:
        vehicle (dict): Vehicle information.
        orders (list): List of orders assigned to the vehicle.
        D (np.ndarray): Distance matrix.
        A (np.ndarray): Adjacency matrix.
        T (np.ndarray): Time matrix.
        params (dict): Simulation parameters.
        charging_strategy (str): Charging strategy ('optimized' or 'no_optimization').

    Returns:
        tuple: (total_rewards, route, charging_events, final_position, final_soc, final_time, passengers)
    """
    # print("accepted vehicle", vehicle['id'], vehicle['active_orders'],)
    env = CarpoolEnv(
        D, A, T, end_time, orders, vehicle, params, charging_strategy=charging_strategy, seed=seed
    )
    agent = QLearning(
        env, lr=params['lr'], eps=params['eps'], eps_min=params['eps_min'],
        eps_decay=params['eps_decay'], gamma=params['gamma'], seed=seed
    )
    agent.run(params['num_episodes'])
    
    route, dist_covered, total_rewards, current_time, passengers, active_orders, charging_events = agent.result()

    # Return vehicle-related values
    final_position = next((pos for pos in reversed(route) if pos < 17), route[0])  # Last position in the route
    final_soc = agent.env.soc  # Final SoC after the trip
    final_time = current_time  # Time after completing all actions
    
    # print("dealed vehicle", vehicle['id'], active_orders)
    return total_rewards, route, charging_events, final_position, final_soc, final_time, passengers, active_orders

