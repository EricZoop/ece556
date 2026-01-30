"""
Genetic Algorithm for evolving neural network weights.
Handles selection, crossover, mutation, and population management.
"""

import random
import json
from neural_network import NeuralNetwork

class GeneticAlgorithm:
    """Manages population of neural networks and evolution process."""
    
    def __init__(self, population_size, input_size=1340, hidden_size=64, 
                 mutation_rate=0.1, mutation_strength=0.3, elite_count=1):
        """
        Args:
            population_size: Number of individuals in population
            input_size: Neural network input size
            hidden_size: Neural network hidden layer size
            mutation_rate: Probability of mutating each weight
            mutation_strength: Magnitude of weight mutations
            elite_count: Number of top performers to keep unchanged
        """
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = elite_count
        
        # Create initial population
        self.population = [
            NeuralNetwork(input_size, hidden_size) 
            for _ in range(population_size)
        ]
        
        # Track fitness scores
        self.fitness_scores = [0.0] * population_size
        
        # Statistics
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def evaluate_population(self, fitness_scores):
        """
        Update fitness scores for current population.
        
        Args:
            fitness_scores: List of fitness values for each individual
        """
        self.fitness_scores = fitness_scores.copy()
        
        # Track statistics
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
    
    def select_parent(self):
        """
        Tournament selection: pick best of 3 random individuals.
        
        Returns:
            Index of selected parent
        """
        tournament_size = 3
        candidates = random.sample(range(self.population_size), tournament_size)
        best_idx = max(candidates, key=lambda i: self.fitness_scores[i])
        return best_idx
    
    def crossover(self, parent1_idx, parent2_idx):
        """
        Single-point crossover between two parents.
        
        Args:
            parent1_idx: Index of first parent
            parent2_idx: Index of second parent
        
        Returns:
            New NeuralNetwork child
        """
        parent1 = self.population[parent1_idx]
        parent2 = self.population[parent2_idx]
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Single-point crossover
        crossover_point = random.randint(0, len(weights1) - 1)
        child_weights = weights1[:crossover_point] + weights2[crossover_point:]
        
        # Create child network
        child = NeuralNetwork(self.input_size, self.hidden_size)
        child.set_weights(child_weights)
        
        return child
    
    def mutate(self, network):
        """
        Apply random mutations to network weights.
        
        Args:
            network: NeuralNetwork to mutate
        """
        weights = network.get_weights()
        
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                # Add Gaussian noise
                weights[i] += random.gauss(0, self.mutation_strength)
        
        network.set_weights(weights)
    
    def evolve(self):
        """
        Create next generation using selection, crossover, and mutation.
        """
        # Sort population by fitness
        sorted_indices = sorted(range(self.population_size), 
                              key=lambda i: self.fitness_scores[i], 
                              reverse=True)
        
        new_population = []
        
        # Elitism: Keep top performers unchanged
        for i in range(self.elite_count):
            elite_idx = sorted_indices[i]
            new_population.append(self.population[elite_idx].clone())
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            parent1_idx = self.select_parent()
            parent2_idx = self.select_parent()
            
            child = self.crossover(parent1_idx, parent2_idx)
            self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def get_best_network(self):
        """
        Get the neural network with highest fitness.
        
        Returns:
            Tuple of (best_network, best_fitness, best_index)
        """
        best_idx = max(range(self.population_size), 
                      key=lambda i: self.fitness_scores[i])
        return self.population[best_idx], self.fitness_scores[best_idx], best_idx
    
    def save_to_file(self, filename):
        """
        Save entire population and statistics to JSON file.
        
        Args:
            filename: Path to save file
        """
        data = {
            'generation': self.generation,
            'population_size': self.population_size,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
            'elite_count': self.elite_count,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'fitness_scores': self.fitness_scores,
            'population': [net.save_to_dict() for net in self.population]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_best_network(self, filename):
        """
        Save only the best performing network.
        
        Args:
            filename: Path to save file
        """
        best_net, best_fitness, best_idx = self.get_best_network()
        
        data = {
            'generation': self.generation,
            'fitness': best_fitness,
            'network': best_net.save_to_dict()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_from_file(filename):
        """
        Load population from JSON file.
        
        Args:
            filename: Path to load file
        
        Returns:
            GeneticAlgorithm instance
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        ga = GeneticAlgorithm(
            population_size=data['population_size'],
            input_size=data['input_size'],
            hidden_size=data['hidden_size'],
            mutation_rate=data['mutation_rate'],
            mutation_strength=data['mutation_strength'],
            elite_count=data['elite_count']
        )
        
        # Restore state
        ga.generation = data['generation']
        ga.best_fitness_history = data['best_fitness_history']
        ga.avg_fitness_history = data['avg_fitness_history']
        ga.fitness_scores = data['fitness_scores']
        
        # Restore population
        ga.population = [
            NeuralNetwork.load_from_dict(net_data) 
            for net_data in data['population']
        ]
        
        return ga
    
    @staticmethod
    def load_best_network(filename):
        """
        Load a single best network from file.
        
        Args:
            filename: Path to load file
        
        Returns:
            NeuralNetwork instance
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return NeuralNetwork.load_from_dict(data['network'])
    
    def get_statistics(self):
        """
        Get current generation statistics.
        
        Returns:
            Dictionary with stats
        """
        if not self.fitness_scores:
            return {
                'generation': self.generation,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'worst_fitness': 0.0
            }
        
        return {
            'generation': self.generation,
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': sum(self.fitness_scores) / len(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores)
        }
