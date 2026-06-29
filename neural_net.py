

import csv
import math


class NeuralNetwork:
    def __init__(self, weight_file):
        self.layers = [4, 8, 8, 2]
        self.weights = self.load_weights(weight_file)

    def load_weights(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                for val in row:
                    data.append(float(val))

        expected = 4 * 8 + 8 * 8 + 8 * 2
        if len(data) != expected:
            raise ValueError(f"Expected {expected} weights, got {len(data)}")

        weights = []
        idx = 0

        shapes = [(4, 8), (8, 8), (8, 2)]

        for (inp, out) in shapes:
            matrix = []
            for i in range(inp):
                row = data[idx:idx + out]
                matrix.append(row)
                idx += out
            weights.append(matrix)

        return weights

    def relu(self, x):
        return max(0.0, x)

    def forward_layer(self, inputs, weight_matrix):
        outputs = []
        for j in range(len(weight_matrix[0])):  # each output neuron
            s = 0.0
            for i in range(len(inputs)):
                s += inputs[i] * weight_matrix[i][j]
            outputs.append(self.relu(s))
        return outputs

    def forward(self, inputs):
        if len(inputs) != 4:
            raise ValueError("Input must have 4 values")

        x = inputs
        for w in self.weights[:-1]:
            x = self.forward_layer(x, w)

        # Last layer (no ReLU, raw output)
        final_weights = self.weights[-1]
        outputs = []
        for j in range(len(final_weights[0])):
            s = 0.0
            for i in range(len(x)):
                s += x[i] * final_weights[i][j]
            outputs.append(s)

        return outputs


if __name__ == "__main__":
    nn = NeuralNetwork("weights.csv")  # construct the Neural Network

    # Example input
    sample_input = [0.5, -0.2, 0.1, 0.9]

    output = nn.forward(sample_input)  # Evaluate the Neural Network. It is your controller

    print("Output:", output)