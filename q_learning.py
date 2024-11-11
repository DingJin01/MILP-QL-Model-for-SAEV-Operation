import numpy as np
from itertools import count
import logging
import random

# Configure logger for this module
logger = logging.getLogger(__name__)

class QLearning:
    def __init__(self, env, lr=0.05, eps=1.0, eps_min=0.1, eps_decay=0.0000001, gamma=0.8, q_value_threshold=5e-2, patience=100, seed=None):
        self.env = env
        self.lr = lr
        self.eps = eps
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.num_episodes = 0
        self.Q = np.zeros([env.n_positions, env.soc_levels, env.n_actions])
        self.q_value_threshold = q_value_threshold  # Threshold for Q-value convergence
        self.patience = patience  # Number of episodes to wait after Q-values converge
        self.no_improvement_count = 0  # Count episodes without significant Q-value improvement
        self.max_q_value_change = float('inf')  # Track the maximum Q-value change in an episode
        self.seed = seed  # Add seed parameter

        if seed is not None:
            np.random.seed(seed)  # Set seed for numpy
            random.seed(seed)  # Set seed for random module

    def run(self, num_episodes, log_step=0, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Reset seed at the start of training
            random.seed(seed)

        if num_episodes <= 0:
            return

        if self.eps_min is None or self.eps_min > self.eps or self.eps_decay == 0:
            self.eps_min = self.eps
        eps_max = self.eps

        self.num_episodes += num_episodes

        for i_ep in range(num_episodes):
            position, soc, valid_actions = self.env.reset()
            max_q_value_change_in_episode = 0.0  # Reset Q-value change tracking for each episode

            for t in count():
                if len(valid_actions) == 0:
                    break

                # Epsilon-greedy action selection
                if np.random.uniform(0, 1) < self.eps:
                    action = self.env.sample()
                else:
                    q_values = self.Q[position, soc, valid_actions]
                    action = valid_actions[np.argmax(q_values)]

                # Step environment and get reward, next state
                new_state, reward, done, valid_actions, current_time, passengers, active_orders = self.env.step(action)

                # Get the maximum Q-value for the next state
                max_q_new_state = np.max(self.Q[new_state[0], new_state[1], valid_actions]) if len(valid_actions) > 0 else 0

                # Calculate the Q-value update
                old_q_value = self.Q[position, soc, action]
                new_q_value = old_q_value + self.lr * (reward + self.gamma * max_q_new_state - old_q_value)
                self.Q[position, soc, action] = new_q_value

                # Track the maximum Q-value change in this episode
                q_value_change = abs(new_q_value - old_q_value)
                max_q_value_change_in_episode = max(max_q_value_change_in_episode, q_value_change)

                # Update state
                position, soc = new_state

                if done:
                    break

            # Update epsilon (exploration rate)
            self.eps = self.eps_min + (eps_max - self.eps_min) * np.exp(-self.eps_decay * (self.num_episodes - num_episodes + i_ep))

            # Update the global maximum Q-value change
            self.max_q_value_change = max_q_value_change_in_episode

            # Check for Q-value convergence
            if self.max_q_value_change < self.q_value_threshold:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0  # Reset count if Q-values changed significantly

            # If no significant change for `patience` episodes, stop training
            if self.no_improvement_count >= self.patience:
                converged_episode = i_ep + 1
                # print(f"Q-values converged at episode {converged_episode}.")
                break
        else:
            # Executed only if the loop didn't encounter a 'break'
            print(f"Q-values did not converge after {num_episodes} episodes.")

    def result(self):
        # print("Received active_orders at the start of result:", self.env.active_orders)  # Initial check
        # Evaluate the learned policy by running it once
        position, soc, valid_actions = self.env.reset()
        rewards = 0.0

        for t in count():
            valid_actions = np.asarray(valid_actions)
            if len(valid_actions) == 0:
                break

            # Select the best action based on learned Q-values
            q_values = self.Q[position, soc, valid_actions]
            idx = np.argmax(q_values)
            action = valid_actions[idx]

            # Step environment and collect reward
            new_state, reward, done, valid_actions, current_time, passengers, active_orders = self.env.step(action)
            
            self.env.active_orders = active_orders
            

            # print("self.env.active_orders", self.env.active_orders)

            position, soc = new_state
            rewards += reward
            if done:
                break

        # print("current_time", current_time)
        # print("active_orders", active_orders)
        # print("done", done)

        # Combine finished and active orders for reporting
        all_orders = self.env.finished_orders + self.env.active_orders
        return self.env.route, self.env.dist_covered, rewards, self.env.current_time, self.env.passengers, self.env.active_orders, self.env.charging_events
