"""
Genetic Algorithm for evolving neural network weights.
Handles selection, crossover, mutation, and population management.
"""

import random
import json
from neural_network import NeuralNetwork


class GeneticAlgorithm:
    """Manages population of neural networks and evolution process."""

    def __init__(self, population_size, input_size=33, hidden_size=32, output_size=4,
                 mutation_rate=0.15, mutation_strength=0.3, elite_count=2):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = elite_count

        # Adaptive mutation parameters
        self.initial_mutation_rate = mutation_rate
        self.initial_mutation_strength = mutation_strength
        self.generations_without_improvement = 0
        self.last_best_fitness = 0

        # Create initial population
        self.population = [
            NeuralNetwork(input_size, hidden_size, output_size)
            for _ in range(population_size)
        ]

        self.fitness_scores = [0.0] * population_size

        # Statistics
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def evaluate_population(self, fitness_scores):
        self.fitness_scores = fitness_scores.copy()

        if not fitness_scores:
            return

        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        if best_fitness > self.last_best_fitness + 50:
            self.generations_without_improvement = 0
            self.last_best_fitness = best_fitness
        else:
            self.generations_without_improvement += 1

        if self.generations_without_improvement > 3:
            self.mutation_rate = min(0.4, self.initial_mutation_rate * 2.0)
            self.mutation_strength = min(0.6, self.initial_mutation_strength * 2.0)
        else:
            self.mutation_rate = self.initial_mutation_rate
            self.mutation_strength = self.initial_mutation_strength

    def select_parent(self):
        """
        Tournament selection with population-scaled tournament size.
        For N=6:  tournament=2  (33% of pop — enough pressure, keeps diversity)
        For N=10: tournament=3  (30% of pop)
        For N=20: tournament=4  (20% of pop)
        """
        tournament_size = max(2, min(4, self.population_size // 3))
        candidates = random.sample(range(self.population_size), tournament_size)
        best_idx = max(candidates, key=lambda i: self.fitness_scores[i])
        return best_idx

    def crossover(self, parent1_idx, parent2_idx):
        parent1 = self.population[parent1_idx]
        parent2 = self.population[parent2_idx]

        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()

        if len(weights1) > 2:
            point1 = random.randint(0, len(weights1) - 2)
            point2 = random.randint(point1 + 1, len(weights1) - 1)

            child_weights = (weights1[:point1] +
                             weights2[point1:point2] +
                             weights1[point2:])
        else:
            mid = len(weights1) // 2
            child_weights = weights1[:mid] + weights2[mid:]

        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        child.set_weights(child_weights)

        return child

    def mutate(self, network):
        weights = network.get_weights()

        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                mutation = random.gauss(0, self.mutation_strength)
                weights[i] += mutation
                weights[i] = max(-5.0, min(5.0, weights[i]))

        network.set_weights(weights)

    def evolve(self):
        """
        Create next generation.

        How elite_count works:
        - The top `elite_count` networks are cloned UNCHANGED into the
          next generation (indices 0..elite_count-1).
        - The remaining (population_size - elite_count) slots are filled
          by crossover + mutation of tournament-selected parents.
        - Elites CAN be selected as parents for children, so their DNA
          spreads, but their own copy is never mutated.

        With 6 bots and elite_count=2:
          Slots 0-1: exact clones of Gen N's best two
          Slots 2-5: children bred from the full population (including elites)
        """
        sorted_indices = sorted(
            range(self.population_size),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )

        new_population = []

        # Elitism: top performers survive unchanged
        for i in range(self.elite_count):
            elite_idx = sorted_indices[i]
            new_population.append(self.population[elite_idx].clone())

        # Breed the rest
        while len(new_population) < self.population_size:
            parent1_idx = self.select_parent()
            parent2_idx = self.select_parent()

            while parent2_idx == parent1_idx and self.population_size > 1:
                parent2_idx = self.select_parent()

            child = self.crossover(parent1_idx, parent2_idx)
            self.mutate(child)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def get_best_network(self):
        best_idx = max(range(self.population_size),
                       key=lambda i: self.fitness_scores[i])
        return self.population[best_idx], self.fitness_scores[best_idx], best_idx

    def save_to_file(self, filename):
        data = {
            'generation': self.generation,
            'population_size': self.population_size,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
            'elite_count': self.elite_count,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'fitness_scores': self.fitness_scores,
            'population': [net.save_to_dict() for net in self.population],
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def save_best_network(self, filename):
        best_net, best_fitness, best_idx = self.get_best_network()

        data = {
            'generation': self.generation,
            'fitness': best_fitness,
            'network': best_net.save_to_dict(),
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        output_size = data.get('output_size', 4)

        ga = GeneticAlgorithm(
            population_size=data['population_size'],
            input_size=data['input_size'],
            hidden_size=data['hidden_size'],
            output_size=output_size,
            mutation_rate=data['mutation_rate'],
            mutation_strength=data['mutation_strength'],
            elite_count=data['elite_count'],
        )

        ga.generation = data['generation']
        ga.best_fitness_history = data['best_fitness_history']
        ga.avg_fitness_history = data['avg_fitness_history']
        ga.fitness_scores = data['fitness_scores']

        ga.population = [
            NeuralNetwork.load_from_dict(net_data)
            for net_data in data['population']
        ]

        return ga

    @staticmethod
    def load_best_network(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return NeuralNetwork.load_from_dict(data['network'])

    def get_statistics(self):
        if not self.fitness_scores:
            return {
                'generation': self.generation,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'worst_fitness': 0.0,
                'mutation_rate': self.mutation_rate,
                'mutation_strength': self.mutation_strength,
            }

        return {
            'generation': self.generation,
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': sum(self.fitness_scores) / len(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores),
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
        }