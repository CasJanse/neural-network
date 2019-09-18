import inline as inline
import numpy as np
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot


def main():
    input_nodes_amount = 784
    hidden_nodes_amount = 100
    output_nodes_amount = 10
    learning_rate = 0.3
    NN = NeuralNetwork(input_nodes_amount, hidden_nodes_amount, output_nodes_amount, learning_rate)

    # Load number image training data
    data_file = open("mnist_train_100.csv")
    data_list = data_file.readlines()
    data_file.close()

    # Train the Neural Network
    for record in data_list:
        # Split the records at every comma
        all_values = record.split(',')

        # Scale input to range 0.01 to 1.00
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # Create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes_amount) + 0.01

        # all_values[0] is the target value for this record
        targets[int(all_values[0])] = 0.99
        print(targets)
        NN.train(inputs, targets)

    # Test the Neural Network
    test_data_file = open("mnist_test_10.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    all_values = test_data_list[0].split(',')
    print(all_values[0])
    print(NN.query(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)


main()

