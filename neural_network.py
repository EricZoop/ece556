"""
Pure-Python feedforward neural network for parkour bot decision making.
v3: Mixed activations - sigmoid for boolean outputs, tanh for continuous yaw.
"""

import math
import json
import random
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _tanh(x):
    return math.tanh(max(-20.0, min(20.0, x)))


class NeuralNetwork:
    """
    Single-hidden-layer net: input →[tanh]→ hidden →[mixed]→ outputs.

    Output activations:
        indices 0..output_size-2  → sigmoid → bool (> 0.5)
        index   output_size-1     → tanh    → float in [-1, +1]

    For the parkour bot:  [strafe_left, strafe_right, jump, yaw_delta]
    yaw_delta of -1 = full left turn, +1 = full right turn.
    """

    def __init__(self, input_size=33, hidden_size=32, output_size=4):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden  = [[0.0] * hidden_size for _ in range(input_size)]
        self.bias_hidden           = [0.0] * hidden_size
        self.weights_hidden_output = [[0.0] * output_size for _ in range(hidden_size)]
        self.bias_output           = [0.0] * output_size

        self._randomize_weights()

    # ------------------------------------------------------------------
    def _randomize_weights(self):
        xi = math.sqrt(2.0 / (self.input_size  + self.hidden_size))
        xo = math.sqrt(2.0 / (self.hidden_size + self.output_size))

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] = (random.random() - 0.5) * 2 * xi

        for j in range(self.hidden_size):
            self.bias_hidden[j] = (random.random() - 0.5) * 0.1

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] = (random.random() - 0.5) * 2 * xo

        for j in range(self.output_size):
            self.bias_output[j] = (random.random() - 0.5) * 0.1

    # ------------------------------------------------------------------
    def _pad_inputs(self, inputs):
        n = self.input_size
        if len(inputs) < n:
            return list(inputs) + [0.0] * (n - len(inputs))
        return inputs[:n]

    # ------------------------------------------------------------------
    def forward(self, inputs):
        """
        Returns:
            list - first N-1 elements are bool, last element is float in [-1, +1]
        """
        inputs = self._pad_inputs(inputs)

        # Hidden layer: tanh activation
        hidden = []
        for j in range(self.hidden_size):
            a = self.bias_hidden[j]
            for i in range(self.input_size):
                a += inputs[i] * self.weights_input_hidden[i][j]
            hidden.append(_tanh(a))

        # Output layer: compute raw pre-activation values
        raw = []
        for j in range(self.output_size):
            a = self.bias_output[j]
            for i in range(self.hidden_size):
                a += hidden[i] * self.weights_hidden_output[i][j]
            raw.append(a)

        # Mixed activations:
        #   indices 0..N-2 → sigmoid → bool
        #   index   N-1    → tanh    → continuous [-1, +1]
        actions = []
        for j in range(self.output_size - 1):
            actions.append(_sigmoid(raw[j]) > 0.5)
        actions.append(_tanh(raw[-1]))  # yaw_delta: -1 to +1

        return actions

    # ------------------------------------------------------------------
    def forward_with_activations(self, inputs):
        inputs = self._pad_inputs(inputs)

        hidden_act = []
        for j in range(self.hidden_size):
            a = self.bias_hidden[j]
            for i in range(self.input_size):
                a += inputs[i] * self.weights_input_hidden[i][j]
            hidden_act.append(_tanh(a))

        raw = []
        for j in range(self.output_size):
            a = self.bias_output[j]
            for i in range(self.hidden_size):
                a += hidden_act[i] * self.weights_hidden_output[i][j]
            raw.append(a)

        # Final activations for logging/visualization
        output_act = []
        for j in range(self.output_size - 1):
            output_act.append(_sigmoid(raw[j]))
        output_act.append(_tanh(raw[-1]))

        actions = []
        for j in range(self.output_size - 1):
            actions.append(output_act[j] > 0.5)
        actions.append(output_act[-1])

        activations = {
            'input':  np.array(inputs,      dtype=np.float32),
            'hidden': np.array(hidden_act,  dtype=np.float32),
            'output': np.array(output_act,  dtype=np.float32),
        }
        return actions, activations

    # ------------------------------------------------------------------
    def get_weights(self):
        w = []
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                w.append(self.weights_input_hidden[i][j])
        w.extend(self.bias_hidden)
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                w.append(self.weights_hidden_output[i][j])
        w.extend(self.bias_output)
        return w

    def set_weights(self, weights):
        idx = 0
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] = weights[idx]; idx += 1
        for j in range(self.hidden_size):
            self.bias_hidden[j] = weights[idx]; idx += 1
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] = weights[idx]; idx += 1
        for j in range(self.output_size):
            self.bias_output[j] = weights[idx]; idx += 1

    # ------------------------------------------------------------------
    def save_to_dict(self):
        return {
            'input_size':  self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'weights':     self.get_weights(),
        }

    @staticmethod
    def load_from_dict(data):
        net = NeuralNetwork(
            input_size  = data['input_size'],
            hidden_size = data['hidden_size'],
            output_size = data['output_size'],
        )
        net.set_weights(data['weights'])
        return net

    def clone(self):
        new_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_net.set_weights(self.get_weights())
        return new_net