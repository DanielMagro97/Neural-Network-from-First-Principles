import numpy as np          # for matrix multiplication, e
import matplotlib
import matplotlib.pyplot as plt
from typing import List     # for type annotation

# seeds the randomness for repeatable results, this can be commented out
np.random.seed(0)


# method which computes sigmoid for any input value
def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


# read the binary function which is stored as a CSV File
with open('BinaryFunction.csv', 'r', encoding='utf-8') as f:
    data = [line.strip().split(',') for line in f.read().strip().split('\n')]
    # remove the first row, containing the header of each column
    data.pop(0)
    data = [[int(j) for j in i] for i in data]

# Randomising the order of the data
np.random.shuffle(data)
# Splitting the data into a training set (26) and a test set (6)
training_data = data[0:26]
test_data = data[26:32]


# Hyper Parameters
# maximum number of epochs the learning algorithm should run for (if convergence is not reached)
max_number_of_epochs: int = 1000
# number of nodes that will be used in the hidden layer
nodes_in_hidden_layer = 4
# the error threshold
error_threshold = 0.2
# the learning rate
learning_rate = 0.2

# initialising a matrix representing the weights of the hidden layer as a 5 x (no. of nodes in hidden layer) matrix
# to a matrix of small random numbers
hidden_layer_weights = np.random.normal(0, 0.1, (5, nodes_in_hidden_layer))
# initialising a matrix representing the weights of the output layer as a (no. of nodes in hidden layer) x 3 matrix
# to a matrix of small random numbers
output_layer_weights = np.random.normal(0, 0.1, (nodes_in_hidden_layer, 3))

# storing the percentage of bad facts per epoch in a list of floats
percentage_of_bad_facts_per_epoch: List[float] = list()

number_of_epochs: int = 0
# Do the training process until the termination condition is met
# Termination condition: Run training until either The Number of Bad Facts is 0
# or some maximum number of epochs have been executed
while True:
    # store the number of bad facts encountered during this epoch
    number_of_bad_facts: int = 0

    # Run all the training instances through the neural net every epoch
    for j in range(len(training_data)):
        # separate the first 5 elements of a training instance (input) into a separate 1x5 matrix
        input = [training_data[j][0:5]]
        # separate the last 3 elements of a training instance (target) into a separate 1x3 matrix
        target = [training_data[j][5:8]]

        # calculate the net of the hidden layer as being the matrix multiplication of the input and the weights of the
        # hidden layer. result is a 1x4 matrix
        hidden_layer_net = np.matmul(input, hidden_layer_weights)
        # compute sigmoid on the net of the hidden layer. result remains a 1x4 matrix
        hidden_layer_output = np.zeros((1, nodes_in_hidden_layer))
        for k in range(len(hidden_layer_net[0])):
            hidden_layer_output[0][k] = sigmoid(hidden_layer_net[0][k])

        # calculate the net of the output layer as being the matrix multiplication of the output of the hidden layer
        # and the weights of the output layer. result is a 1x3 matrix
        output_layer_net = np.matmul(hidden_layer_output, output_layer_weights)
        # compute sigmoid on the net of the output layer. result remains a 1x3 matrix
        output_layer_output = np.zeros((1, 3))
        for k in range(len(output_layer_net[0])):
            output_layer_output[0][k] = sigmoid(output_layer_net[0][k])

        # compute the error
        error = target - output_layer_output

        # check if the error threshold has been exceeded by any of the bits.
        threshold_exceeded: bool = False
        for k in error[0]:
            if abs(k) > error_threshold:
                threshold_exceeded = True
                # increment the number of bad facts
                number_of_bad_facts += 1
                break

        if threshold_exceeded:
            # Do Error Back Propagation

            # compute 𝛿 of the each neuron in the output layer. There should be 3 values
            output_layer_deltas = np.zeros((1, 3))
            for k in range(len(output_layer_deltas[0])):
                # set 𝛿 of node k in the output layer to:
                output_layer_deltas[0][k] = output_layer_output[0][k] \
                                            * (1 - output_layer_output[0][k]) \
                                            * (target[0][k] - output_layer_output[0][k])
            # Adjust the weights of the output layer
            for k in range(len(output_layer_weights)):
                for l in range(len(output_layer_weights[k])):
                    # set Δw_kl_o to the learning rate * 𝛿 of node l in the output layer
                    # * the output of node k in the hidden layer
                    delta_kl_o = learning_rate * output_layer_deltas[0][l] * hidden_layer_output[0][k]

                    # Updating the weight of the connection from node k in the hidden layer
                    # to node l in the output layer
                    output_layer_weights[k][l] += delta_kl_o


            # compute 𝛿 of the each neuron in the hidden layer. There should be 4 values
            hidden_layer_deltas = np.zeros((1, nodes_in_hidden_layer))
            for k in range(len(hidden_layer_deltas[0])):
                # loop over the nodes that are downstream of the current node in the hidden layer to calculate sigma
                sigma: float = 0
                for l in range(len(output_layer_weights[k])):
                    sigma += output_layer_weights[k][l] * output_layer_deltas[0][l]

                # set 𝛿 of node k in the hidden layer to:
                hidden_layer_deltas[0][k] = hidden_layer_output[0][k] \
                                            * (1 - hidden_layer_output[0][k]) \
                                            * sigma
            # Adjust the weights of the hidden layer
            for k in range(len(hidden_layer_weights)):
                for l in range(len(hidden_layer_weights[k])):
                    # set Δw_kl_h to the learning rate * 𝛿_l_h * the output of node k in the input layer (i.e. the input)
                    delta_kl_h = learning_rate * hidden_layer_deltas[0][l] * input[0][k]

                    # Updating the weight of the connection from node k in the input layer
                    # to node l in the hidden layer
                    hidden_layer_weights[k][l] += delta_kl_h

    # store the percentage of bad facts found during this epoch
    percentage_of_bad_facts_per_epoch.append((number_of_bad_facts / len(training_data)) * 100)
    # increment the number of epochs
    number_of_epochs += 1

    # Termination condition: Run training until either The Number of Bad Facts is 0
    # or some maximum number of epochs have been executed
    if number_of_epochs >= max_number_of_epochs:
        print("Neural Network has not converged after the max number of epochs (" + str(max_number_of_epochs) + " epochs), stopping training now.")
        break
    if percentage_of_bad_facts_per_epoch[-1] == 0:
        break

# Calculating test error (Performance)
test_good_facts: int = 0
for i in range(len(test_data)):
    # separate the first 5 elements of a test instance (input) into a separate 1x5 matrix
    input = [test_data[i][0:5]]
    # separate the last 3 elements of a test instance (target) into a separate 1x3 matrix
    target = [test_data[i][5:8]]

    # calculate the net of the hidden layer as being the matrix multiplication of the input
    # and the weights of the hidden layer. result is a 1x4 matrix
    hidden_layer_net = np.matmul(input, hidden_layer_weights)
    # compute sigmoid on the net of the hidden layer. result remains a 1x4 matrix
    hidden_layer_output = np.zeros((1, nodes_in_hidden_layer))
    for k in range(nodes_in_hidden_layer):
        hidden_layer_output[0][k] = sigmoid(hidden_layer_net[0][k])

    # calculate the net of the output layer as being the matrix multiplication of the output of the hidden layer
    # and the weights of the output layer. result is a 1x3 matrix
    output_layer_net = np.matmul(hidden_layer_output, output_layer_weights)
    # compute sigmoid on the net of the output layer. result remains a 1x3 matrix
    output_layer_output = np.zeros((1, 3))
    for k in range(len(output_layer_net[0])):
        output_layer_output[0][k] = sigmoid(output_layer_net[0][k])

    # since the target is made up of bits (0 or 1)
    # and the outputs of the Neural Net will be floats in the range [0,1]
    # values >= 0.5 in the output are set to 1, whereas values < 0.5 are set to 0.
    output_layer_output[output_layer_output >= 0.5] = 1
    output_layer_output[output_layer_output < 0.5] = 0

    # the output is then compared to the target. If all bits are equal, it is counted as a good fact
    if np.array_equal(output_layer_output, target):
        test_good_facts += 1

# The performance is the number of good facts over the total number of test instances
print("Number of Epochs: " + str(number_of_epochs))
performance: float = (test_good_facts/len(test_data)) * 100
print(f"Performance: {performance.__format__('.2f')}%")

# Plot the Percentage of Bad Facts against Epochs Graph
# the x-values of each point are the epoch in which each percentage of bad facts was obtained. (0, 1, 2, 3, ...)
# the y-values are the percentage of bad facts during that epoch. (eg. 100%, 96%, 92%, ..., 4%, 0%)
x = np.arange(len(percentage_of_bad_facts_per_epoch))
plt.plot(x, percentage_of_bad_facts_per_epoch, color='blue', linestyle='-')
plt.title("Percentage of Bad Facts per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Bad Facts")
plt.xlim(left=0)
plt.xlim(right=number_of_epochs)
plt.ylim(bottom=0)
plt.ylim(top=100)
plt.show()
