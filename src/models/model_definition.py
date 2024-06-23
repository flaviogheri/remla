'''
This file contains the model definition.
The model is a Convolutional Neural Network (CNN) with multiple Conv1D layers.
'''

# src/models/model_definition.py

import os
import tensorflow as tf
# print(tf.__version__)
# import keras 
# print (keras.__version__)

from keras.api.models import Sequential
from keras.api.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

import yaml

project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_file = os.path.join(project_directory, "config.yml")
# print(config_file)
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
params = config['params']


def build_model():
    '''
    This function builds the model.
    '''
    model = Sequential()
    voc_size = config['params']['char_index_size']
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(len(params['categories'])-1, activation='sigmoid'))

    return model
