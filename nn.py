import numpy as np

# Neural network model
# Weights and biases for propagation to the hidden layer
w_0 = np.array([[5, 1],
                [9, 3],
                [4, 8]])
b_0 = np.array([[0,],
                [4,],
                [9,]])

# Weights and biases for propagation to the output layer
w_1 = np.array([[1, 4, 0],
                [3, 9, 2]])
b_1 = np.array([[2,],
                [7,]])

# Inputs to the neural network
inputs = np.array( [[2,],
                    [3,]])

hidden_layer = w_0 @ inputs + b_0
#hidden_layer = np.matmul(w_0, inputs) + b_0

output_layer = w_1 @ hidden_layer + b_1

#Print out the output layer
print(output_layer)

