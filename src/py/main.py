import numpy as np
from NeuralNetwork import NeuralNetwork


def main():
    # Set amount of nodes in each layer
    input_nodes_amount = 784
    hidden_nodes_amount = 100
    output_nodes_amount = 10
    learning_rate = 0.3

    # Create the neural network
    NN = NeuralNetwork(input_nodes_amount, hidden_nodes_amount, output_nodes_amount, learning_rate)

    # Use previously trained network or train a new one
    load_weights = input("Load weights from previous networks? (y/n)")
    if load_weights.lower()[0] == "y":
        NN.load_weights()
    else:
        # Load number image training data (60.000 records)
        train_data_file = open("../csv/mnist_train.csv")
        train_data_list = train_data_file.readlines()
        train_data_file.close()

        # Train the Neural Network
        for record in train_data_list:
            # Split the records at every comma
            all_values = record.split(',')

            # Scale input to range 0.01 to 1.00
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # Create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes_amount) + 0.01

            # all_values[0] is the target value for this record
            targets[int(all_values[0])] = 0.99
            NN.train(inputs, targets)

    # Test the Neural Network (10.000 records)
    test_data_file = open("../csv/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # Test the Neural Network
    # Scorecard for how well the network performs, initially empty
    scorecard = []

    # Go through all the records in the test data set
    for record in test_data_list:
        # Split the record at every comma
        all_values = record.split(',')

        # Correct answer in first value
        correct_label = int(all_values[0])

        # Scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # Query the network
        outputs = NN.query(inputs)

        # The index of the highest value corresponds to the label
        label = np.argmax(outputs)

        # Append correct or incorrect to list
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print(scorecard)

    # Calculate the performance score
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)

    save_weights = " "
    while save_weights.lower()[0] != "y" and save_weights.lower()[0] != "n":
        save_weights = input("Save current network weights to file? (y/n)")
        if save_weights.lower()[0] == "y":
            NN.save_weights()
            print('saved weights')


main()

