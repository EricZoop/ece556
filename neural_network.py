"""
Simple feedforward neural network for parkour bot decision making.
Uses sensory inputs to generate movement actions.
"""

import math
import json

class NeuralNetwork:
    """Feedforward neural network with one hidden layer for bot control."""
    
    def __init__(self, input_size=1340, hidden_size=64, output_size=6):
        """
        Args:
            input_size: Size of flattened voxel grid + other sensors
            hidden_size: Number of neurons in hidden layer
            output_size: 6 actions [forward, back, left, right, jump, sprint]
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values
        self.weights_input_hidden = [[0.0] * hidden_size for _ in range(input_size)]
        self.bias_hidden = [0.0] * hidden_size
        self.weights_hidden_output = [[0.0] * output_size for _ in range(hidden_size)]
        self.bias_output = [0.0] * output_size
        
        # Randomize initial weights
        self.randomize_weights()
    
    def randomize_weights(self, scale=0.5):
        """Initialize weights with random values."""
        import random
        
        # Input to hidden
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] = (random.random() - 0.5) * scale
        
        for j in range(self.hidden_size):
            self.bias_hidden[j] = (random.random() - 0.5) * scale
        
        # Hidden to output
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] = (random.random() - 0.5) * scale
        
        for j in range(self.output_size):
            self.bias_output[j] = (random.random() - 0.5) * scale
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))  # Clamp to prevent overflow
    
    def tanh(self, x):
        """Tanh activation function."""
        return math.tanh(max(-20, min(20, x)))
    
    def forward(self, inputs):
        """
        Forward pass through the network.
        
        Args:
            inputs: List of sensor values (flattened)
        
        Returns:
            List of 6 boolean actions [forward, back, left, right, jump, sprint]
        """
        # Ensure input is the correct size
        if len(inputs) != self.input_size:
            # Pad or truncate
            if len(inputs) < self.input_size:
                inputs = inputs + [0.0] * (self.input_size - len(inputs))
            else:
                inputs = inputs[:self.input_size]
        
        # Hidden layer
        hidden = []
        for j in range(self.hidden_size):
            activation = self.bias_hidden[j]
            for i in range(self.input_size):
                activation += inputs[i] * self.weights_input_hidden[i][j]
            hidden.append(self.tanh(activation))
        
        # Output layer
        outputs = []
        for j in range(self.output_size):
            activation = self.bias_output[j]
            for i in range(self.hidden_size):
                activation += hidden[i] * self.weights_hidden_output[i][j]
            outputs.append(self.sigmoid(activation))
        
        # Convert to boolean actions (threshold at 0.5)
        actions = [output > 0.5 for output in outputs]
        
        return actions
    
    def get_weights(self):
        """Return all weights as a flat list for genetic algorithm."""
        weights = []
        
        # Flatten input->hidden weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                weights.append(self.weights_input_hidden[i][j])
        
        # Add hidden biases
        weights.extend(self.bias_hidden)
        
        # Flatten hidden->output weights
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                weights.append(self.weights_hidden_output[i][j])
        
        # Add output biases
        weights.extend(self.bias_output)
        
        return weights
    
    def set_weights(self, weights):
        """Set all weights from a flat list."""
        idx = 0
        
        # Set input->hidden weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] = weights[idx]
                idx += 1
        
        # Set hidden biases
        for j in range(self.hidden_size):
            self.bias_hidden[j] = weights[idx]
            idx += 1
        
        # Set hidden->output weights
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] = weights[idx]
                idx += 1
        
        # Set output biases
        for j in range(self.output_size):
            self.bias_output[j] = weights[idx]
            idx += 1
    
    def save_to_dict(self):
        """Save network structure and weights to a dictionary."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'weights': self.get_weights()
        }
    
    @staticmethod
    def load_from_dict(data):
        """Load network from a dictionary."""
        network = NeuralNetwork(
            input_size=data['input_size'],
            hidden_size=data['hidden_size'],
            output_size=data['output_size']
        )
        network.set_weights(data['weights'])
        return network
    
    def clone(self):
        """Create a copy of this network."""
        new_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_net.set_weights(self.get_weights())
        return new_net
