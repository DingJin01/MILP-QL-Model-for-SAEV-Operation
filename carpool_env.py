import numpy as np
import copy
import math
from itertools import count
import logging
import random

class CarpoolEnv:
    def __init__(self, D, A, T, interval_end_time, new_orders, vehicle, params, charging_strategy, seed=None):
        self.D = D
        self.A = A  # Adjacency matrix to define valid actions
        self.T = T
        self.n_positions = D.shape[0]
        self.charging_strategy = charging_strategy
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)  # Set seed for numpy
            random.seed(seed)  # Set seed for random module

        if self.charging_strategy == 'optimized':
            self.n_actions = D.shape[1] + 11  # Added one more action for waiting
        elif self.charging_strategy == 'no_optimization':
            self.n_actions = D.shape[1] + 1  # Added one more action for waiting
        else:
            raise ValueError("Charging strategy must be 'optimized' or 'no_optimization'.")

        self.interval_end_time = interval_end_time
        self.initial_new_orders = copy.deepcopy(new_orders)
        self.finished_orders = []
        self.initial_position = vehicle['position']
        self.initial_time = vehicle['time']
        self.initial_passengers = vehicle['passengers']
        self.initial_active_orders = copy.deepcopy(vehicle['active_orders'])
        self.initial_soc = vehicle['soc']
        self.distance_threshold = params['order_pickup_radius']
        self.order_delay_threshold = params['order_delay_threshold']
        self.capacity = params['capacity']
        self.price_per_distance = params['price_per_distance']
        self.cost_per_distance = params['cost_per_distance']
        self.soc_levels = params['soc_levels']
        self.electricity_price = 0.1 * params['electricity_price']
        self.battery_capacity = params['battery_capacity']
        self.KWh_per_km = params['KWh_per_km']
        self.arrival_rates = params['arrival_rates']
        self.service_rate = params['service_rate']
        self.server = params['server']
        self.unit_waiting_cost = params['unit_waiting_cost']

        self.position = self.initial_position
        self.current_time = self.initial_time
        self.passengers = self.initial_passengers
        self.wait_orders = self.filter_orders(self.initial_new_orders)
        self.active_orders = copy.deepcopy(self.initial_active_orders)

        self.done = False
        self.dist_covered = 0.0
        self.cumulated_reward = 0.0

        self.route = [self.position]
        self.valid_actions = self.get_valid_actions(self.initial_position, self.initial_soc)

        self.soc = self.initial_soc
        self.charging = False
        self.charging_events = []

    def get_valid_actions(self, position, soc):
        # Valid movement actions: nodes connected to the current position
        valid_moves = [i for i, connected in enumerate(self.A[position]) if connected == 1 and i not in self.route]
        
        # Ensure the movement action is possible with the available SoC
        valid_moves = [i for i in valid_moves if self.D[position][i] <= soc]
        
        # Valid charging actions: increments of 10% up to a max SoC of 100%
        if soc < 100 and self.charging_strategy == 'optimized':
            # Filter charging actions to ensure SoC does not exceed 100%
            valid_charge = [self.n_positions + (i // 10) for i in range(10, 101, 10) if soc + i <= 100]
        else:
            valid_charge = []

        # Add wait action
        wait_action = [self.n_positions]

        # Combine movement, charging, and wait actions
        valid_actions = valid_moves + valid_charge + wait_action
        return valid_actions

    def calculate_Wq(self, lambda_rate, mu_rate, c):
        # Calculate rho (utilization factor)
        if c == 0:
            return 100
        
        rho = lambda_rate / (c * mu_rate)
        
        if rho >= 1:
            return 100  # Indicate that the queue is unstable
        
        # Calculate P0 (probability of 0 customers in the system)
        sum_terms = sum((rho * c)**n / math.factorial(n) for n in range(c))
        P0 = sum_terms + ((rho * c)**c / math.factorial(c)) * (1 / (1 - rho))
        P0 = P0**(-1)
        
        # Calculate Lq (average number of customers in the queue)
        Lq = (P0 * (rho * c)**c * rho) / (math.factorial(c) * (1 - rho)**2)
        
        # Calculate Wq (average waiting time in the queue)
        Wq = Lq / lambda_rate
        
        return Wq + (1/60)

    def filter_orders(self, orders):
        return [order for order in orders if self.D[order['start'], self.initial_position] <= self.distance_threshold]

    def charge_time(self, start_soc, charge_soc):
        start_soc = start_soc / 100
        charge_soc = charge_soc / 100

        def T(soc):
            if 0 <= soc < 0.05:
                return (4 / 5) * soc
            elif 0.05 <= soc < 0.1:
                return (2 / 125) * np.log(50 * np.exp(5 / 2) * (soc - 0.03))
            elif 0.1 <= soc <= 1:
                return -(36 / 175) * np.log(-(10 / 9) * ((7 / 2) ** (-7 / 90)) * np.exp(-7 / 36) * (soc - 1.01))
            else:
                raise ValueError("SoC must be between 0 and 1")

        end_soc = start_soc + charge_soc

        if not (0 <= start_soc <= 1) or not (0 <= end_soc <= 1):
            raise ValueError("SoC must be between 0 and 1")

        charging_time = T(end_soc) - T(start_soc)

        return charging_time

    def reset(self):
        self.position = self.initial_position
        self.current_time = self.initial_time
        self.passengers = self.initial_passengers
        self.soc = self.initial_soc
        self.wait_orders = self.filter_orders(self.initial_new_orders)
        self.active_orders = copy.deepcopy(self.initial_active_orders)
        self.finished_orders = []
        self.done = False
        self.dist_covered = 0.0
        self.cumulated_reward = 0.0
        self.route = [self.position]
        self.valid_actions = self.get_valid_actions(self.initial_position, self.initial_soc)
        self.charging = False
        self.charging_events = []

        return self.position, self.soc, self.valid_actions

    def sample(self):
        action = np.random.choice(self.valid_actions)
        return action

    def step(self, action):
        reward = 0.0
        self.done = self.current_time >= self.interval_end_time
        if self.done:
            # Exit early if the time interval has ended
            return (self.position, self.soc), reward, self.done, self.valid_actions, self.current_time, self.passengers, self.active_orders

        self.action = action
        # Handle the action based on its type (driving, charging, or waiting)
        if self.charging_strategy == 'optimized':
            if self.action >= self.n_positions + 1:
                # Charging event
                reward += self.handle_optimized_charging_event()
            elif self.action == self.n_positions:
                # Wait action
                reward += self.handle_wait_event()
            else:
                # Driving event
                reward += self.handle_driving_event()
        elif self.charging_strategy == 'no_optimization':
            reward += self.handle_rule_based_charging_event()
            if self.done:
                # Exit early if the time interval has ended
                return (self.position, self.soc), reward, self.done, self.valid_actions, self.current_time, self.passengers, self.active_orders            
            if self.action == self.n_positions:
                # Wait action
                reward += self.handle_wait_event()
            else:
                # Driving event
                reward += self.handle_driving_event()
        else:
            raise ValueError("Charging strategy must be 'optimized' or 'no_optimization'.")

        # Check for excessive delay for all active orders and apply penalties if needed
        reward += self.check_delay_and_force_drop_off(self.order_delay_threshold)

        # Update the list of valid actions
        self.valid_actions = self.get_valid_actions(self.position, self.soc)

        return (self.position, self.soc), reward, self.done, self.valid_actions, self.current_time, self.passengers, self.active_orders

    def handle_wait_event(self):
        reward = 0.0
        waiting_time = self.interval_end_time - self.current_time

        # Apply waiting cost
        waiting_cost = self.passengers * waiting_time * self.unit_waiting_cost
        reward -= waiting_cost

        # Update current time to the end of the interval
        self.route.append(self.action)
        self.current_time = self.interval_end_time
        self.done = True

        return reward

    def handle_optimized_charging_event(self):
        reward = 0.0
        self.charging = True
        self.charging_events.append(f"Vehicle is charging. Current time: {self.current_time}, SoC: {self.soc}")
        
        # Calculate the actual charging amount (since action is encoded as an index)
        self.charge_soc = (self.action - self.n_positions) * 10  # Ensure a minimum of 10% charge increment

        num_servers = self.server[self.position]  # Ensure c is a single integer
        self.charging_time = self.charge_time(self.soc, self.charge_soc) + self.calculate_Wq(self.arrival_rates[self.position], self.service_rate, num_servers)
        initial_time = self.current_time
        self.current_time += self.charging_time
        self.soc = min(self.soc + self.charge_soc, 100)  # Cap SoC at 100%

        # Calculate costs
        charging_cost = self.electricity_price[self.position, int(initial_time)] * self.battery_capacity * self.charge_soc / 100
        charging_benefit = np.mean(self.electricity_price) * self.battery_capacity * self.charge_soc / 100
        waiting_cost = self.passengers * self.charging_time * self.unit_waiting_cost

        reward = charging_benefit - charging_cost - waiting_cost
        self.cumulated_reward += reward
        self.route.append(self.action)
        self.charging = False
        self.charging_events.append(f"Vehicle charging break. Current time: {self.current_time}, SoC: {self.soc}")

        self.done = self.current_time >= self.interval_end_time

        return reward

    def handle_rule_based_charging_event(self):
        reward = 0.0
        target_soc = 60  # Charge to 70%

        if self.soc <= 40:
            self.charging = True
            self.charging_events.append(f"Vehicle is charging. Current time: {self.current_time}, SoC: {self.soc}")
            self.charge_soc = target_soc - self.soc

            num_servers = self.server[self.position]  # Ensure c is a single integer
            self.charging_time = self.charge_time(self.soc, self.charge_soc) + self.calculate_Wq(self.arrival_rates[self.position], self.service_rate, num_servers)
            # print(self.position)
            # print(self.soc)
            # print(self.charge_time(self.soc, self.charge_soc))
            # print(self.calculate_Wq(self.arrival_rates[self.position], self.service_rate, num_servers))

            initial_time = self.current_time
            self.current_time += self.charging_time
            self.soc += self.charge_soc

            # Calculate costs
            charging_cost = self.electricity_price[self.position, int(initial_time)] * self.battery_capacity * self.charge_soc / 100
            charging_benefit = np.mean(self.electricity_price) * self.battery_capacity * self.charge_soc / 100
            waiting_cost = (self.passengers + 1) * self.charging_time * self.unit_waiting_cost

            reward = charging_benefit - charging_cost - waiting_cost
            self.cumulated_reward += reward
            self.route.append(18)
            self.charging = False
            self.charging_events.append(f"Vehicle charging break. Current time: {self.current_time}, SoC: {self.soc}")

            self.done = self.current_time >= self.interval_end_time

        return reward

    def check_delay_and_force_drop_off(self, delay_threshold):
        penalty = 0.0
        for order in self.active_orders[:]:
            T_direct = self.T[order['start'], order['goal']]
            T_consumed = self.current_time - order['pick_up_time']

            if T_consumed > (delay_threshold + 1) * T_direct:
                fare = 0.5 * order['passengers'] * self.D[order['start'], order['goal']] * self.price_per_distance
                penalty += -2 * fare  # Apply double fare penalty

                self.active_orders.remove(order)
                self.finished_orders.append(order)
                self.passengers -= order['passengers']

        return penalty

    def handle_driving_event(self):
        reward = 0.0
        distance = self.D[self.position, self.action]
        travel_time = self.T[self.position, self.action]

        for order in self.wait_orders[:]:
            if order['start'] == self.position and order['passengers'] <= self.capacity - self.passengers:
                reward += self.handle_pick_up(order)

        self.current_time += travel_time
        self.dist_covered += distance
        self.soc -= distance
        self.route.append(self.action)
        
        reward -= self.cost_per_distance * distance  # Operational cost for driving

        for order in self.active_orders[:]:
            D_before = self.D[self.position, order['goal']]
            D_after = self.D[self.action, order['goal']]
            D_step = self.D[self.position, self.action]
            fare = 0.5 * order['passengers'] * self.D[order['start'], order['goal']] * self.price_per_distance
            order_reward = ((D_before - D_after)/(self.dist_covered + D_after))*fare
            reward += order_reward

            if order['goal'] == self.action:
                reward += self.handle_drop_off(order)

        for order in self.wait_orders[:]:
            if order['start'] == self.action and order['passengers'] <= self.capacity - self.passengers:
                reward += self.handle_pick_up(order)

        self.position = self.action
        self.cumulated_reward += reward

        # if not self.wait_orders and not self.active_orders:
        #     self.done = True
        if self.current_time >= self.interval_end_time:
            self.done = True

        return reward

    def handle_drop_off(self, order):
        order_delay = (self.current_time - order['pick_up_time'] - self.T[order['start'], order['goal']])/(self.T[order['start'], order['goal']])
        reward = (0.25 * order['passengers'] * self.D[order['start'], order['goal']] * self.price_per_distance
                  if order_delay <= (1 + self.order_delay_threshold)
                  else -1 * order['passengers'] * self.D[order['start'], order['goal']] * self.price_per_distance)
        self.passengers -= order['passengers']
        self.active_orders.remove(order)
        self.finished_orders.append(order)  # Record the finished order
        return reward

    def handle_pick_up(self, order):
        reward = 0.25 * order['passengers'] * self.D[order['start'], order['goal']] * self.price_per_distance
        order['pick_up_time'] = self.current_time
        self.passengers += order['passengers']
        self.active_orders.append(order)
        self.wait_orders.remove(order)
        return reward
