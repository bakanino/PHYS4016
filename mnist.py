import keras
from keras.layers import InputLayer, Dense
import numpy as np
from tensorflow.random import set_seed
set_seed(0)

npz_file = np.load('mnist_1-3.npz')
for _ in npz_file.files:
    print(_)
    print(npz_file[_])
x_train = npz_file['x_train']
y_train = npz_file['y_train']
x_test = npz_file['x_test']
y_test = npz_file['y_test']
npz_file.close()

model = keras.Sequential()

model.add(InputLayer(28**2))
model.add(Dense(15, activation='relu'))
model.add(Dense(3, activation='softmax'))

#model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy) #0.9836323857307434
