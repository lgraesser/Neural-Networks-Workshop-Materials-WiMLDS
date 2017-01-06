import pickle
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

# Load data and plot some examples
train_x = pickle.load(open("MNIST_train_x.pkl", 'rb'))
train_y = pickle.load(open("MNIST_train_y.pkl", 'rb'))
test_x = pickle.load(open("MNIST_test_x.pkl", 'rb'))
test_y = pickle.load(open("MNIST_test_y.pkl", 'rb'))

print(type(train_x))
print(train_x.shape)
print(type(train_y))
print(train_y.shape)
print(type(test_x))
print(test_x.shape)
print(type(train_y))
print(train_y.shape)

# Creating smaller training dataset to speed up training
train_x_short = train_x[:20000]
train_y_short = train_y[:20000]

# A first model in keras
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, verbose=2, validation_split=0.2)
print()

# Printing model information to screen
model.summary()
print()

# Example of recompiling a keras model
model2 = Sequential()
model2.add(Dense(128, input_dim=784))
model2.add(Activation('sigmoid'))
model2.add(Dense(10))
model2.add(Activation('sigmoid'))

print("Learning rate = 0.1")
sgd = SGD(lr=0.1)
model2.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model2.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, verbose=2, validation_split=0.2)
print()
print("Learning rate = 0.01")
sgd = SGD(lr=0.01)
model2.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model2.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, verbose=2, validation_split=0.2)
print()
print("Learning rate = 0.001")
sgd = SGD(lr=0.001)
model2.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model2.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, verbose=2, validation_split=0.2)

