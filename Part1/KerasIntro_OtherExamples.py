import pickle
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

train_x = pickle.load(open("MNIST_train_x.pkl", 'rb'))
train_y = pickle.load(open("MNIST_train_y.pkl", 'rb'))
test_x = pickle.load(open("MNIST_test_x.pkl", 'rb'))
test_y = pickle.load(open("MNIST_test_y.pkl", 'rb'))
train_x_short = train_x[:20000]
train_y_short = train_y[:20000]

print("Categorical crossentropy vs. quadratic cost")
### Quadratic cost (mean squared error) vs. categorical crossentropy
# - Categorical cross-entropy significantly speeds up training
# - Softmax output layers are the most appropriate for the MNIST problem since each image can only 
#   belong to one class and softmax outputs a proability distribution across the 10 classes.
#     - As the value of one output node increases, the value of one or more other output nodes must decrease
#     - This is consistent with our intuition that as we become more confident and image belongs to one class, 
#       we reduce our confidence that an image belongs to other classes

# Softmax output layer, mse
print("Quadratic cost (slow, poorer learning)")
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=128, nb_epoch=10, validation_split=0.2, verbose=2)
print()

# Softmax output layer, categorical crossentropy
print("Categorical crossentropy, (fast, quick learning)")
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=128, nb_epoch=10, validation_split=0.2, verbose=2)
print()

print("Testing batch sizes")
# Reducing the batch_size tends to increase the amount learnt per epoch, but also increases time to complete an epoch
# - In the experiments below total time to reach a comparable accuracy level was broadly similar
# - Reducing batch size from 32 to 16 appeared to hurt performance

print("Batch size = 128")
start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=128, nb_epoch=10, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))
print()

print("Batch size = 64")
start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=64, nb_epoch=7, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))
print()

print("Batch size = 32")
start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=6, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))
print()

print("Batch size = 16")
start = time.time()
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=16, nb_epoch=6, validation_split=0.2, verbose=2)
end = time.time()
print("Model took  {} seconds to complete".format(end - start))
print()

print("Testing Relu")
# Relu + softmax
# - Needs a low learning rate for the network to learn anything
# - Performs worse than a sigmoid hidden layer for shallow networks

print("Relu hidden layer, 3 layer network")
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)
print()

print("Sigmoid hidden layer, 3 layer network")
model2 = Sequential()
model2.add(Dense(128, input_dim=784))
model2.add(Activation('sigmoid'))
model2.add(Dense(10))
model2.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)
print()

# Relu really comes into its own for deep networks
# - Deeper network tend to perform better than shallow networks for complex tasks
# - But they are hard to train. Relu's make it easier for deep networks to learn because their 
#   gradients don't saturate for postive inputs

print("Relu hidden layers, 5 layer network")
model3 = Sequential()
model3.add(Dense(512, input_dim=784))
model3.add(Activation('relu'))
model3.add(Dense(256))
model3.add(Activation('relu'))
model3.add(Dense(128))
model3.add(Activation('relu'))
model3.add(Dense(64))
model3.add(Activation('relu'))
model3.add(Dense(10))
model3.add(Activation('softmax'))

sgd = SGD(lr=0.001)
model3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)


print("Sigmoid hidden layers, 3 layer network")
model3 = Sequential()
model3.add(Dense(512, input_dim=784))
model3.add(Activation('sigmoid'))
model3.add(Dense(256))
model3.add(Activation('sigmoid'))
model3.add(Dense(128))
model3.add(Activation('sigmoid'))
model3.add(Dense(64))
model3.add(Activation('sigmoid'))
model3.add(Dense(10))
model3.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(train_x_short, train_y_short, batch_size=32, nb_epoch=10, validation_split=0.2, verbose=2)

