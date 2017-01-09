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
'''
First step is to initialize your model
Keras has two models 
    1. Sequential - easier to work with, suitable for most tasks
    2. Functional AP - useful for defining complex models
'''
model = Sequential()

'''Then, define your model architecture'''

'''
Add a fully connected hidden layer with 100 nodes. 
When you add your first layer, Keras implicitly adds the input layer, 
so you need to specify the dimension of your inputs
'''
model.add(Dense(100, input_dim=784))
'''Specify your activation function for this layer'''
model.add(Activation('sigmoid'))
'''Add an output layer with 10 output nodes'''
model.add(Dense(10))
'''Specify your activation function for this layer'''
model.add(Activation('sigmoid'))

'''
Next, compile your model. This defines two critical features
    1. Optimizer - how your model learns
    2. Loss function - how your model defines the error between
       the correct output and its prediction
Here you can also specify the metrics you want to use to evaluate your model's performance
'''
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

'''
Finally, train your model. 
To train a model needs input data, and the corresponding correct outputs
You can also specify the batch size and number of training epochs
'''
model.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, verbose=2)
print()

# Printing model information to screen
model.summary()
print()

'''
# Example of recompiling a keras model
   - This allows you to change the settings of the optimizer and loss function if you wish,
     without affecting the values of the weights and biases
   - It can be useful for reducing the learning rate if your model performance has plateaued
   - This example also illustrates the validation_split option of the fit function. 
     It holds out a specified proportion of your training data for evaluating model 
     performance as it trains. 
      - This enables you to tune your hyperparameters and model architecture without polluting 
        the test data
'''
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

# Helper function to calculate model accuracy on the test data
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    num_correct = np.argmax(result, axis=1)==np.argmax(test_y, axis=1)
    accuracy = np.sum(num_correct) / result.shape[0]
    print("Accuracy on data is: {}%".format(accuracy * 100))

accuracy(test_x, test_y, model)
