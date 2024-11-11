# genetic_algorithm.py

import random
import copy
import itertools
from utils import calculate_rewards  # Now correctly imported

class GeneticAlgorithm:
    def __init__(self, vehicles, orders, D, A, T, params, charging_strategy, pop_size=50, generations=100, mutation_rate=0.1):
        self.vehicles = vehicles
        self.orders = orders
        self.D = D
        self.A = A
        self.T = T
        self.params = params
        self.charging_strategy = charging_strategy
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            chromosome = self.create_balanced_chromosome()
            population.append(chromosome)
        return population

    def create_balanced_chromosome(self):
        chromosome = {vehicle['id']: [] for vehicle in self.vehicles}
        orders_copy = copy.deepcopy(self.orders)
        random.shuffle(orders_copy)

        vehicle_cycle = itertools.cycle(self.vehicles)
        for order in orders_copy:
            assigned_vehicle = next(vehicle_cycle)
            chromosome[assigned_vehicle['id']].append(order)

        return chromosome

    def fitness(self, chromosome):
        total_rewards = 0
        for vehicle in self.vehicles:
            vehicle_orders = chromosome[vehicle['id']]
            if vehicle_orders:
                rewards, _, _ = calculate_rewards(vehicle, vehicle_orders, self.D, self.A, self.T, self.params, self.charging_strategy)
                total_rewards += rewards
        return total_rewards

    def selection(self):
        # Sort population based on fitness
        sorted_population = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)
        # Select top 50%
        return sorted_population[:self.pop_size // 2]

    def crossover(self, parent1, parent2):
        child1, child2 = {}, {}
        crossover_point = len(self.vehicles) // 2
        vehicles1 = list(parent1.keys())[:crossover_point]
        vehicles2 = list(parent1.keys())[crossover_point:]

        for v in vehicles1:
            child1[v] = copy.deepcopy(parent1[v])
            child2[v] = copy.deepcopy(parent2[v])
        for v in vehicles2:
            child1[v] = copy.deepcopy(parent2[v])
            child2[v] = copy.deepcopy(parent1[v])

        # Ensure no order is duplicated between vehicles
        for child in [child1, child2]:
            all_orders = set()
            for v in child:
                unique_orders = []
                for order in child[v]:
                    if order['id'] not in all_orders:
                        unique_orders.append(order)
                        all_orders.add(order['id'])
                child[v] = unique_orders

        return child1, child2

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            vehicle1, vehicle2 = random.sample(list(chromosome.keys()), 2)
            if chromosome[vehicle1]:
                order = random.choice(chromosome[vehicle1])
                chromosome[vehicle1].remove(order)
                chromosome[vehicle2].append(order)
        return chromosome

    def evolve(self):
        for generation in range(self.generations):
            selected_population = self.selection()
            next_generation = []
            while len(next_generation) < self.pop_size:
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                next_generation.extend([child1, child2])
            self.population = next_generation[:self.pop_size]

            if (generation + 1) % 10 == 0 or generation == 0:
                best_fitness = self.fitness(max(self.population, key=lambda x: self.fitness(x)))
                print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        best_chromosome = max(self.population, key=lambda x: self.fitness(x))
        best_fitness = self.fitness(best_chromosome)
        return best_chromosome, best_fitness
