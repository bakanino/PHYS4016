import matplotlib.pyplot as plt
import keras
from keras.layers import InputLayer, Conv2D, Flatten, Dense
import numpy as np
from tensorflow.random import set_seed
set_seed(0)

npz_file = np.load('mnist_1-3.npz')
for _ in npz_file.files:
    print(_)
    print(npz_file[_])
x_train = npz_file['x_train'].reshape((-1, 28, 28, 1))
y_train = npz_file['y_train']
x_test = npz_file['x_test'].reshape((-1, 28, 28, 1))
y_test = npz_file['y_test']
npz_file.close()

model = keras.Sequential()
print(y_train.shape)
print(x_train.shape)

model.add(InputLayer(x_train.shape[1:]))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(15, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

#model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10)
#
loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy)
#Neural Network: 0.9864652156829834
#Convolutional Neural Network: 0.9915013909339905

predictions = model.predict(x_test)
examples = 100
print('Predictions:', np.argmax(predictions[:examples], axis=1))
print('Ground truth:', np.argmax(y_test[:examples], axis=1))
print('Equality test:', np.argmax(predictions[:examples], axis=1) == np.argmax(y_test[:examples], axis=1))
for i in range(x_test.shape[0]):
    predicted = np.argmax(predictions[i])
    truth = np.argmax(y_test[i])
    if predicted != truth:
        print('Wrong prediction on test point', i)
        plt.figure()
        plt.imshow(x_test[i].reshape((28, 28)), cmap='binary')
        plt.title('Predicted = {}, Truth = {}'.format(predicted+1, truth+1))
        plt.savefig('im/Test point {}.png'.format(i))
        plt.close()
