import unittest
import pytest

class InstallTest(unittest.TestCase):

    def keras_import(self):
        import sys
        import keras
        assert('keras' in sys.modules)

    def simple_nn(self):
        import numpy as np
        from keras.models import Sequential
        from keras.layers.core import Dense, Activation
        from keras.optimizers import SGD

        model = Sequential()
        model.add(Dense(2, input_dim=4))
        model.add(Activation('sigmoid'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        sgd = SGD(lr=0.01)
        model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
        result = model.predict(np.asarray([1,2,3,4]).reshape([1, 4]))
        assert(type(result) is np.ndarray)



