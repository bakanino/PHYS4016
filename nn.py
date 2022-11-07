import keras
from keras.layers import InputLayer, Dense
import numpy as np

# Weights and biases for propagation to the hidden layer
w_0 = np.array([[5, 1],
                [9, 3],
                [4, 8]]).T
b_0 = np.array([0, 4, 9])

# Weights and biases for propagation to the output layer
w_1 = np.array([[1, 4, 0],
                [3, 9, 2]]).T
b_1 = np.array([2, 7])

# Inputs to the neural network
inputs = np.array( [[2,],
                    [3,]]).T

#hidden_layer = w_0 @ inputs + b_0
##hidden_layer = np.matmul(w_0, inputs) + b_0
#
#output_layer = w_1 @ hidden_layer + b_1

model = keras.Sequential()

# Add the layers
model.add(InputLayer(2))
model.add(Dense(3, weights = [w_0, b_0]))
model.add(Dense(2, weights = [w_1, b_1]))

# Predict and print
print(model.predict(inputs))
