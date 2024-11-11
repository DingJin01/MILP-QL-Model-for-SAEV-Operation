import gurobipy as gp
from gurobipy import GRB
from carpool_env import CarpoolEnv
from q_learning import QLearning
from utils import filter_orders_for_vehicle, calculate_rewards
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
from copy import deepcopy

def evaluate_combination_for_milp(combination, vehicle, D, A, T, params, charging_strategy, order_threshold, distance_threshold, interval_end_time):
    if len(combination) > order_threshold:
        return None

    # Check the additional distance constraint for orders with the same start location
    for order1, order2 in itertools.combinations(combination, 2):
        if order1['start'] == order2['start']:
            distance_order1 = D[order1['start'], order1['goal']]
            distance_order2 = D[order2['start'], order2['goal']]
            distance_between_goals = D[order1['goal'], order2['goal']]
            detour_rate_order1 = (distance_order2 + distance_between_goals - distance_order1)/distance_order1
            detour_rate_order2 = (distance_order1 + distance_between_goals - distance_order2)/distance_order2
            if min(detour_rate_order1,detour_rate_order2) > distance_threshold:
                return None

    # print("Input calculate_rewards", vehicle['active_orders'])
    total_rewards, route, charging_events, final_position, final_soc, final_time, passengers, active_orders = calculate_rewards(
        vehicle, list(combination), D, A, T, params, charging_strategy, interval_end_time)
    order_ids = tuple(sorted(order['id'] for order in combination))
    return order_ids, total_rewards, vehicle['id'], route, charging_events

def solve_vehicle_order_matching_parallel(vehicles, orders, D, A, T, params, charging_strategy, order_threshold, distance_threshold, interval_end_time):
    # Debug statement to verify vehicle data type
    if not isinstance(vehicles, list) or not all(isinstance(vehicle, dict) for vehicle in vehicles):
        raise TypeError("vehicles should be a list of dictionaries.")

    if not isinstance(orders, list):
        raise TypeError("orders should be a list of dictionaries.")
    
    total_order_count = len(orders)
    print(f"Total number of orders in this interval: {total_order_count}")
    
    order_to_vehicle_rewards = {}
    order_to_vehicle_routes = {}
    order_to_vehicle_charging = {}

    # Use ProcessPoolExecutor for parallelization (multiprocessing)
    with ProcessPoolExecutor() as executor:
        # List to hold futures for parallel execution
        futures = []

        for vehicle in vehicles:
            # print("parallel vehicle:", vehicle)
            if 'position' not in vehicle:
                raise KeyError(f"Vehicle dictionary missing 'position' key: {vehicle}")

            feasible_orders = filter_orders_for_vehicle(vehicle, orders, D, params['order_pickup_radius'])

            # Case with no orders for this vehicle (empty combination)
            futures.append(
                executor.submit(evaluate_combination_for_milp, [], deepcopy(vehicle), D, A, T, params, charging_strategy, order_threshold, distance_threshold, interval_end_time)
            )

            # Parallelize the combinations for this vehicle using processes
            for r in range(1, len(feasible_orders) + 1):
                for combination in itertools.combinations(feasible_orders, r):
                    # print("vehicle", vehicle, "combination", combination)
                    # Submit each combination evaluation as a parallel task using processes
                    futures.append(
                        executor.submit(evaluate_combination_for_milp, deepcopy(combination), deepcopy(vehicle), D, A, T, params, charging_strategy, order_threshold, distance_threshold, interval_end_time)
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
                # print(order_ids, vehicle_id, reward)

        # Constraints: Each order can be served by at most one vehicle
        for order in orders:
            model.addConstr(sum(x[(order_ids, vehicle_id)]
                                for order_ids, reward_list in order_to_vehicle_rewards.items()
                                if order['id'] in order_ids
                                for reward, vehicle_id, route in reward_list) <= 1,
                            f'order_{order["id"]}_constraint')

        # Constraints: Each vehicle can serve at most one set of orders
        for vehicle in vehicles:
            model.addConstr(sum(x[(order_ids, vehicle['id'])]
                                for order_ids, reward_list in order_to_vehicle_rewards.items()
                                for reward, vehicle_id, route in reward_list
                                if vehicle_id == vehicle['id']) <= 1,
                            f'vehicle_{vehicle["id"]}_constraint')
        
        # Each vehicle must be assigned at least one combination
        for vehicle in vehicles:
            model.addConstr(
                gp.quicksum(x[(order_ids, vehicle['id'])] for order_ids in order_to_vehicle_rewards if vehicle['id'] in [v_id for _, v_id, _ in order_to_vehicle_rewards[order_ids]]) >= 1,
                f'vehicle_{vehicle["id"]}_must_be_assigned'
            )

        # Set objective to maximize total rewards
        objective = gp.quicksum(reward * x[(order_ids, vehicle_id)]
                                for (order_ids, vehicle_id), var in x.items()
                                for reward, v_id, route in order_to_vehicle_rewards[order_ids]
                                if v_id == vehicle_id)
        model.setObjective(objective, GRB.MAXIMIZE)

        # Optimize the MILP model
        model.optimize()

        results = []

        allocated_orders = set()

        if model.status == GRB.OPTIMAL:
            interval_reward = model.objVal
            for (order_ids, vehicle_id), var in x.items():
                if var.x == 1:
                    # The optimal combination of orders for this vehicle
                    results.append((vehicle_id, order_ids))  # Return vehicle_id and the optimal set of orders
                    allocated_orders.update(order_ids)
            
            allocated_order_count = len(allocated_orders)
            print(f"Number of orders allocated in this interval: {allocated_order_count}")

        return results, interval_reward  # Return the optimal results

    except gp.GurobiError as e:
        print(f'Gurobi Error: {e}')
    except AttributeError as e:
        print(f'Attribute Error: {e}')

# Matching Strategy: MILP
def solve_vehicle_order_matching(vehicles, orders, D, A, T, params, charging_strategy, interval_end_time):
    order_to_vehicle_rewards = {}
    order_to_vehicle_routes = {}
    order_to_vehicle_charging = {}

    for vehicle in vehicles:
        feasible_orders = filter_orders_for_vehicle(vehicle, orders, D, params['order_pickup_radius'])
        for combination in itertools.chain.from_iterable(itertools.combinations(feasible_orders, r) for r in range(1, len(feasible_orders) + 1)):
            total_rewards, route, charging_events, final_position, final_soc, final_time, passengers, activate_orders = calculate_rewards(
                vehicle, list(combination), D, A, T, params, charging_strategy, interval_end_time)            
            order_ids = tuple(sorted(order['id'] for order in combination))
            if order_ids not in order_to_vehicle_rewards:
                order_to_vehicle_rewards[order_ids] = []
                order_to_vehicle_routes[order_ids] = []
                order_to_vehicle_charging[order_ids] = []
            order_to_vehicle_rewards[order_ids].append((total_rewards, vehicle['id'], route))
            order_to_vehicle_routes[order_ids].append(route)
            order_to_vehicle_charging[order_ids].append(charging_events)

    try:
        model = gp.Model('VehicleOrderMatching')
        
        x = {}
        for order_ids, reward_list in order_to_vehicle_rewards.items():
            for reward, vehicle_id, route in reward_list:
                x[(order_ids, vehicle_id)] = model.addVar(vtype=GRB.BINARY, name=f'x[{order_ids},{vehicle_id}]')

        for order in orders:
            model.addConstr(sum(x[(order_ids, vehicle_id)] for order_ids, reward_list in order_to_vehicle_rewards.items() if order['id'] in order_ids for reward, vehicle_id, route in reward_list) <= 1, f'order_{order["id"]}_constraint')

        for vehicle in vehicles:
            model.addConstr(sum(x[(order_ids, vehicle['id'])] for order_ids, reward_list in order_to_vehicle_rewards.items() for reward, vehicle_id, route in reward_list if vehicle_id == vehicle['id']) <= 1, f'vehicle_{vehicle["id"]}_constraint')

        objective = gp.quicksum(reward * x[(order_ids, vehicle_id)] for (order_ids, vehicle_id), var in x.items() for reward, v_id, route in order_to_vehicle_rewards[order_ids] if v_id == vehicle_id)
        model.setObjective(objective, GRB.MAXIMIZE)

        model.optimize()

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