# matching_strategies.py

import itertools
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from q_learning import QLearning
from carpool_env import CarpoolEnv
from utils import filter_orders_for_vehicle, calculate_rewards, create_combinations
import gurobipy as gp
from gurobipy import GRB

def simulate_all_vehicles(vehicles, orders, D, A, T, params, charging_strategy):
    total_rewards = 0
    total_routes = []
    total_charging_events = []
    vehicle_orders = {vehicle['id']: [] for vehicle in vehicles}  # Track all orders for each vehicle
    agents = []

    remaining_orders = copy.deepcopy(orders)

    for vehicle in vehicles:
        env = CarpoolEnv(D, A, T, params['time_interval'], remaining_orders, vehicle, params, charging_strategy)
        agent = QLearning(env, lr=params['lr'], eps=params['eps'], eps_min=params['eps_min'], eps_decay=params['eps_decay'], gamma=params['gamma'])
        agent.run(params['num_episodes'])
        agents.append(agent)
        route, dist_covered, rewards, current_time, passengers, all_orders, charging_events = agent.result()

        # Add all served orders to the respective vehicle's list
        vehicle_orders[vehicle['id']].extend([order['id'] for order in all_orders])

        # Remove served orders from the global order pool
        remaining_orders = [order for order in remaining_orders if order['id'] not in vehicle_orders[vehicle['id']]]

        total_rewards += rewards
        total_routes.append(route)
        total_charging_events.append(charging_events)

    # Print the orders picked by each vehicle (both finished and active)
    for vehicle_id, orders in vehicle_orders.items():
        print(f"Vehicle {vehicle_id} served orders: {orders}")

    return total_rewards, total_routes, total_charging_events, agents

def greedy_matching(vehicles, orders, D, A, T, params, charging_strategy, interval_end_time):
    print("greedy starts")
    total_rewards = 0
    total_matched_orders = {vehicle['id']: [] for vehicle in vehicles}
    total_charging_events = {vehicle['id']: [] for vehicle in vehicles}
    agents = {vehicle['id']: None for vehicle in vehicles}
    vehicle_orders = {vehicle['id']: [] for vehicle in vehicles}  # Track all orders for each vehicle

    remaining_orders = copy.deepcopy(orders)

    for vehicle in vehicles:
        print(vehicle)
        # Apply filter based on pickup radius
        feasible_orders = filter_orders_for_vehicle(vehicle, remaining_orders, D, params['order_pickup_radius'])

        best_combination = None
        best_reward = -float('inf')
        best_route = None
        best_charging_events = None
        best_agent = None

        # Iterate over all combinations of the feasible orders
        for combination in itertools.chain.from_iterable(itertools.combinations(feasible_orders, r) for r in range(1, len(feasible_orders) + 1)):
            print(combination)
            current_orders = list(combination)
            env = CarpoolEnv(D, A, T, interval_end_time, current_orders, vehicle, params, charging_strategy)
            agent = QLearning(env, lr=params['lr'], eps=params['eps'], eps_min=params['eps_min'], eps_decay=params['eps_decay'], gamma=params['gamma'])
            agent.run(params['num_episodes'])
            route, dist_covered, rewards, current_time, passengers, all_orders, charging_events = agent.result()

            if rewards > best_reward:
                best_reward = rewards
                best_combination = combination
                best_route = route
                best_charging_events = charging_events
                best_agent = agent

        if best_combination:
            # Remove orders that were part of the best combination from the remaining orders
            for order in best_combination:
                remaining_orders.remove(order)
                vehicle_orders[vehicle['id']].append(order)

            total_rewards += best_reward
            total_matched_orders[vehicle['id']] = [order for order in best_combination]
            total_charging_events[vehicle['id']] = best_charging_events
            agents[vehicle['id']] = best_agent

    # Print the orders picked by each vehicle
    for vehicle_id, orders in vehicle_orders.items():
        order_ids = [order['id'] for order in orders]
        print(f"Vehicle {vehicle_id} served orders: {order_ids}")

    # Return only the agents that were actually used
    used_agents = [agent for agent in agents.values() if agent is not None]

    return total_rewards, total_matched_orders, total_charging_events, used_agents

def evaluate_combination_for_milp(combination, vehicle, D, A, T, params, charging_strategy, order_threshold, distance_threshold):
    if len(combination) > order_threshold:
        return None

    # Check the additional distance constraint for orders with the same start location
    for order1, order2 in itertools.combinations(combination, 2):
        if order1['start'] == order2['start']:
            distance_order1 = D[order1['start'], order1['goal']]
            distance_order2 = D[order2['start'], order2['goal']]
            distance_between_goals = D[order1['goal'], order2['goal']]
            detour_rate_order1 = (distance_order2 + distance_between_goals - distance_order1) / distance_order1
            detour_rate_order2 = (distance_order1 + distance_between_goals - distance_order2) / distance_order2
            if min(detour_rate_order1, detour_rate_order2) > distance_threshold:
                return None

    total_rewards, route, charging_events = calculate_rewards(vehicle, list(combination), D, A, T, params, charging_strategy)
    order_ids = tuple(sorted(order['id'] for order in combination))
    return order_ids, total_rewards, vehicle['id'], route, charging_events

def solve_vehicle_order_matching_parallel(vehicles, orders, D, A, T, params, charging_strategy, order_threshold, distance_threshold):
    order_to_vehicle_rewards = {}
    order_to_vehicle_routes = {}
    order_to_vehicle_charging = {}

    # Use ProcessPoolExecutor for parallelization (multiprocessing)
    with ProcessPoolExecutor() as executor:
        # List to hold futures for parallel execution
        futures = []

        for vehicle in vehicles:
            feasible_orders = filter_orders_for_vehicle(vehicle, orders, D, params['order_pickup_radius'])
            # Parallelize the combinations for this vehicle using processes
            for combination in create_combinations(feasible_orders, max_combination_size=order_threshold):
                # Submit each combination evaluation as a parallel task using processes
                futures.append(
                    executor.submit(evaluate_combination_for_milp, combination, vehicle, D, A, T, params, charging_strategy, order_threshold, distance_threshold)
                )

        # Process each result as it completes
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                order_ids, total_rewards, vehicle_id, route, charging_events = result

                # Organize results in dictionaries for MILP optimization
                if order_ids not in order_to_vehicle_rewards:
                    order_to_vehicle_rewards[order_ids] = []
                    order_to_vehicle_routes[order_ids] = []
                    order_to_vehicle_charging[order_ids] = []

                order_to_vehicle_rewards[order_ids].append((total_rewards, vehicle_id, route))
                order_to_vehicle_routes[order_ids].append(route)
                order_to_vehicle_charging[order_ids].append(charging_events)

    try:
        # Create Gurobi model for MILP
        model = gp.Model('VehicleOrderMatching')

        # Create decision variables
        x = {}
        for order_ids, reward_list in order_to_vehicle_rewards.items():
            for reward, vehicle_id, route in reward_list:
                x[(order_ids, vehicle_id)] = model.addVar(vtype=GRB.BINARY, name=f'x[{order_ids},{vehicle_id}]')

        # Constraints: Each order can be served by at most one vehicle
        for order in orders:
            model.addConstr(
                gp.quicksum(
                    x[(order_ids, vehicle_id)]
                    for order_ids, reward_list in order_to_vehicle_rewards.items()
                    if order['id'] in order_ids
                    for reward, vehicle_id, route in reward_list
                ) <= 1,
                f'order_{order["id"]}_constraint'
            )

        # Constraints: Each vehicle can serve at most one set of orders
        for vehicle in vehicles:
            model.addConstr(
                gp.quicksum(
                    x[(order_ids, vehicle['id'])]
                    for order_ids, reward_list in order_to_vehicle_rewards.items()
                    for reward, vehicle_id, route in reward_list
                    if vehicle_id == vehicle['id']
                ) <= 1,
                f'vehicle_{vehicle["id"]}_constraint'
            )

        # Set objective to maximize total rewards
        objective = gp.quicksum(
            reward * x[(order_ids, vehicle_id)]
            for (order_ids, vehicle_id), var in x.items()
            for reward, v_id, route in order_to_vehicle_rewards[order_ids]
            if v_id == vehicle_id
        )
        model.setObjective(objective, GRB.MAXIMIZE)

        # Optimize the MILP model
        model.optimize()

        # Check the status of the model and print solution
        if model.status == GRB.OPTIMAL:
            print('Optimal solution found:')
            for (order_ids, vehicle_id), var in x.items():
                if var.x > 0.5:
                    for reward, v_id, route in order_to_vehicle_rewards[order_ids]:
                        if v_id == vehicle_id:
                            print(f'Vehicle {vehicle_id} takes orders {order_ids} with reward {reward} and route {route}')
                            for event in order_to_vehicle_charging[order_ids]:
                                if v_id == vehicle_id:
                                    for e in event:
                                        print(e)
        else:
            print('No optimal solution found.')

    except gp.GurobiError as e:
        print(f'Gurobi Error: {e}')
    except AttributeError as e:
        print(f'Attribute Error: {e}')

def solve_vehicle_order_matching(vehicles, orders, D, A, T, params, charging_strategy):
    # This function can be implemented similarly to the parallel version or removed if not needed
    pass  # Implement if necessary
