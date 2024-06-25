'''
This file contains the model definition.
The model is a Convolutional Neural Network (CNN) with multiple Conv1D layers.
'''

# src/models/model_definition.py

import os

from keras.api.models import Sequential  # pylint: disable=import-error, no-name-in-module
from keras.api.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout  \
    # pylint: disable=import-error, no-name-in-module

import yaml

project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_file = os.path.join(project_directory, "config.yml")
# print(config_file)
with open(config_file, "r", encoding="utf-8") as file:
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
