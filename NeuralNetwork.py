import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # Set the number of nodes in each input, hidden, output layer
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        # Learning rate
        self.lr = learningRate

        # Initiate weights
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # Initialise activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # Convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Calculate signals coming into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate hidden layer outputs with activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals coming into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate output layer outputs with the activation function
        final_outputs = self.activation_function(final_inputs)

        # Calculate the error
        output_errors = targets - final_outputs

        # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # Update the weights for the links between the input layer and the hidden layer
        self.wih = self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        # Convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculate signals coming into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate hidden layer outputs with activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals coming into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate output layer outputs with the activation function
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

