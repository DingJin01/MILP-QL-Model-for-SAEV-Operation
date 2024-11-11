import pandas as pd

# File paths
DISTANCE_FILE = 'distance.csv'
ADJACENCY_FILE = 'adjacency.csv'
ELECTRICITY_PRICE_FILE = 'electricity_price.csv'
ORDERS_FILE = 'orders.csv'

# Simulation parameters
PARAMS = {
    # Training Parameters
    'lr': 0.05,
    'eps': 1.0,
    'eps_min': 0.1,
    'eps_decay': 0.0001,
    'gamma': 0.8,
    'num_episodes': 100000,
    # Time Interval Length
    'time_interval': 0.25,
    # Filter Parameters
    'order_pickup_radius': 4,
    'combination_size_threshold': 3,
    'detour_rate_threshold': 1,
    # Vehicle Parameters
    'vehicle_speed': 30,
    'capacity': 4,
    'cost_per_distance': 0.085,
    'soc_levels': 101,  # SoC levels from 0 to 100
    # Passenger Parameters
    'price_per_distance': 1,
    'order_delay_threshold': 1,
    'unit_waiting_cost': 1,
    # Charging Parameters
    'battery_capacity': 80,
    'KWh_per_km': 1,
    'arrival_rates': [15.15, 18.47, 24.53, 0.44, 92.36, 16.26, 13.89, 5.31, 4.0, 32.42, 9.83, 38.67, 88.4, 47.78, 14.18, 13.53, 56.18],
    'service_rate': 4,
    'server': [4, 10, 8, 4, 29, 45, 4, 4, 4, 41, 4, 23, 77, 29, 12, 37, 100],

}

# Vehicles definition
VEHICLES = [
    {'id': 1, 'position': 11, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 10},
    {'id': 2, 'position': 3, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 50},
    {'id': 3, 'position': 5, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 90},
    {'id': 4, 'position': 12, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 10},
    {'id': 5, 'position': 16, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 50},
    {'id': 6, 'position': 1, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 90},
    {'id': 7, 'position': 9, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 10},
    {'id': 8, 'position': 7, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 50},
    {'id': 9, 'position': 5, 'time': 0, 'passengers': 0, 'active_orders': [], 'soc': 90},
]

# Charging strategies
CHARGING_STRATEGIES = ['optimized', 'no_optimization']
