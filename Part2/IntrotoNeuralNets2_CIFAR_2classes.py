'''
Sample code for Introduction to Neural Networks in Python: Part 2
- Topics covered
    - Regularization (weight penalities (l2, l1), dropout, early stopping)
    - Convolutional networks
    - Other techniques for improving the weight networks learn 
      (momentum / other optimizers, weight initialization, adversarial 
      training, dataset augmentation)
'''

import matplotlib.pyplot as plt
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.regularizers import l2, l1

# Load data
train_data = pickle.load(open("CIFAR_2_train_data.pkl", 'rb'))
train_labels = pickle.load(open("CIFAR_2_train_labels.pkl", 'rb'))
test_data = pickle.load(open("CIFAR_2_test_data.pkl", 'rb'))
test_labels = pickle.load(open("CIFAR_2_test_labels.pkl", 'rb'))
print("Training data shape: {}".format(train_data.shape))
print("Training labels shape: {}".format(train_labels.shape))
print("Test data shape: {}".format(test_data.shape))
print("Test labels shape: {}".format(test_labels.shape))
label_dict = ["airplane", "cat"]
label_dict[0]

# Plot some examples
def plotExamples(data, labels, label_dict):
    plt.figure(figsize=(8,5))
    for i in range(8):
        sub = 241 + i
        ax = plt.subplot(sub)
        index = np.random.randint(0, data.shape[0])
        label_int = np.argmax(labels[index])
        ax.set_title(label_dict[label_int])
        im = data[index]
        im = np.transpose(im, (1, 2, 0))
        plt.imshow(im)
    plt.show()

# NOTE: you will need to close the images when they pop up for the script to progress
plotExamples(train_data, train_labels, label_dict)
plotExamples(test_data, test_labels, label_dict)

'''Feedforward networks'''

# First, flatten data from 3D to 1D
def flatten(data):
    return np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))

train_data_flat = flatten(train_data)
test_data_flat = flatten(test_data)
train_data_flat_short = train_data_flat[:5000]
train_labels_short = train_labels[:5000]
print("Train data: Orig shape: {} New shape {}".format(train_data.shape, train_data_flat.shape))
print("Test data: Orig shape: {} New shape {}".format(test_data.shape, test_data_flat.shape))
print("Short train data: Shape {}".format(train_data_flat_short.shape))
print("Short train labels: Shape {}".format(train_labels_short.shape))

#Two simple models to start with, no regularization
print("A simple feedforward model")
# Try running this a few times with and without additional hidden layer. What happens?
model = Sequential()
model.add(Dense(2048, input_dim=3072, activation='relu'))
model.add(Dense(1024, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
sgd = SGD(lr=0.00001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_flat_short, train_labels_short, batch_size=128, nb_epoch=20, validation_split=0.2, verbose=2)
print()

# Adding more layers and making the network layers larger leads to improved performance 
# on the training set, but overfitting is now an issue
# ~14ppts accuracy difference between training and validation dataset 
print("A feedforward model with more layers")
model = Sequential()
model.add(Dense(2048, input_dim=3072, activation='relu', init='lecun_uniform'))
model.add(Dense(1024, activation='relu',  init='lecun_uniform'))
model.add(Dense(512, activation='relu',  init='lecun_uniform'))
model.add(Dense(256, activation='relu',  init='lecun_uniform'))
model.add(Dense(128, activation='relu',  init='lecun_uniform'))
model.add(Dense(2, activation='softmax',  init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.00001, momentum=0.95, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_flat_short, train_labels_short, batch_size=128, nb_epoch=20, validation_split=0.2, verbose=2)
print()

'''
Regularization Examples
    - L1
    - L2
    - Dropout
'''

'''
L2 weight regularization. What happens when you change the regularization parameter?
'''
print("Model with L2 weight regularization")
# You need to add L1 or L2 regularization to each layer, through the W_regularizer parameter
model = Sequential()
model.add(Dense(2048, input_dim=3072, activation='relu', init='lecun_uniform', W_regularizer=l2(0.01)))
model.add(Dense(1024, activation='relu',  init='lecun_uniform', W_regularizer=l2(0.01)))
model.add(Dense(512, activation='relu',  init='lecun_uniform', W_regularizer=l2(0.01)))
model.add(Dense(256, activation='relu',  init='lecun_uniform', W_regularizer=l2(0.01)))
model.add(Dense(128, activation='relu',  init='lecun_uniform', W_regularizer=l2(0.01)))
model.add(Dense(2, activation='softmax',  init='lecun_uniform', W_regularizer=l2(0.01)))
model.summary()
sgd = SGD(lr=0.00001, momentum=0.95, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_flat_short, train_labels_short, batch_size=128, nb_epoch=20, validation_split=0.2, verbose=2)
print()

'''
L1 weight regularization
'''
print("Model with L1 weight regularization")
model = Sequential()
model.add(Dense(2048, input_dim=3072, activation='relu', init='lecun_uniform', W_regularizer=l1(0.01)))
model.add(Dense(1024, activation='relu',  init='lecun_uniform', W_regularizer=l1(0.01)))
model.add(Dense(512, activation='relu',  init='lecun_uniform', W_regularizer=l1(0.01)))
model.add(Dense(256, activation='relu',  init='lecun_uniform', W_regularizer=l1(0.01)))
model.add(Dense(128, activation='relu',  init='lecun_uniform', W_regularizer=l1(0.01)))
model.add(Dense(2, activation='softmax',  init='lecun_uniform', W_regularizer=l1(0.01)))
model.summary()
sgd = SGD(lr=0.00001, momentum=0.95, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_flat_short, train_labels_short, batch_size=128, nb_epoch=20, validation_split=0.2, verbose=2)
print()

'''
Dropout
- What happens if you increase the proportion of nodes dropped?
- What happens if you apply dropout after every layer?
- Can the models be trained for longer now without overfitting? Do more epochs improve the results?
'''

'''
Dropout is incorporated through a separate layer, the same dimension of the previous layer
Each node in the dropout layer corresponds to a node in the layer below it
The operation that each dropout layer node performs is to either output zero with probability p, 
or pass on the output of the previous layer node unchanged with probability (1-p)
This has the effect of randomly "deleting" nodes from the layer below with probability p each forward pass
during training
'''
print("Model with dropout")
model = Sequential()
model.add(Dense(2048, input_dim=3072, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dense(1024, activation='relu',  init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu',  init='lecun_uniform' , W_constraint=maxnorm(3)))
model.add(Dense(256, activation='relu',  init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu',  init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dense(2, activation='softmax',  init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.00001, momentum=0.95, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_flat_short, train_labels_short, batch_size=128, nb_epoch=20, validation_split=0.2, verbose=2)
print()

print("Model with dropout, increasing the size of the hidden layers")
model = Sequential()
model.add(Dense(3000, input_dim=3072, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dense(2000, activation='relu',  init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu',  init='lecun_uniform' , W_constraint=maxnorm(3)))
model.add(Dense(500, activation='relu',  init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu',  init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dense(2, activation='softmax',  init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.00001, momentum=0.95, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_flat_short, train_labels_short, batch_size=128, nb_epoch=40, validation_split=0.2, verbose=2)
print()

'''
Convolutional Networks
'''

train_data_short = train_data[:5000]

'''
Starting with a simple model
- How reliably does it learn?
- What happens when you run the model a few times?
'''
print("A simple convolutional network")
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32), init='lecun_uniform'))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax', init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.000001, momentum=0.9, decay=0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_short, train_labels_short, batch_size=128, nb_epoch=10, validation_split=0.2)
print()

'''
Adding an extra fully connected layer helps learning and appears to stabilize performance
- Do convolutional networks overfit as much as unregularized feedforward networks? Why / why not?
'''
print("Adding a fully connected layer to convolutional network")
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32), init='lecun_uniform'))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', init='lecun_uniform'))
model.add(Dense(2, activation='softmax', init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.000001, momentum=0.9, decay=0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_short, train_labels_short, batch_size=128, nb_epoch=20, validation_split=0.2)
print()

print("Same network as above, trained on the whole dataset for 40 epochs")
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32), init='lecun_uniform'))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', init='lecun_uniform'))
model.add(Dense(2, activation='softmax', init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.000001, momentum=0.9, decay=0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=128, nb_epoch=40, validation_split=0.2)
print()

'''
Putting everything together
- Convolutional layers
- Pooling
- Dense layers
- Dropout
- Weight contstraints
- Momentum
'''
print("Network that puts everything together, including conv layers, pooling, dense, dropout, weight constraints, momentum")
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32), init='lecun_uniform'))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dense(512, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax', init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.000001, momentum=0.9, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_short, train_labels_short, batch_size=128, nb_epoch=20, validation_split=0.2)
print()

print("Same network as above, trained on the whole dataset for 40 epochs")
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32), init='lecun_uniform'))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dense(512, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax', init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.000001, momentum=0.9, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=128, nb_epoch=20, validation_split=0.2)
print()

print("Same network as above, trained on the whole dataset for 60 epochs")
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32), init='lecun_uniform'))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='lecun_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dense(512, activation='relu', init='lecun_uniform', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax', init='lecun_uniform'))
model.summary()
sgd = SGD(lr=0.000001, momentum=0.9, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=128, nb_epoch=40, validation_split=0.2)
print()

print("Continuing training on last network")
sgd = SGD(lr=0.000001, momentum=0.9, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=128, nb_epoch=20, validation_split=0.2)