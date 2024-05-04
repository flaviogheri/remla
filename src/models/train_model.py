
import numpy as np
import yaml
from model_defintion import build_model

# Load parameters and data
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

params = config['params']


# UKNOWN HERE : Do I load data using np.load or something else !?
x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

# Build the model
voc_size = params['char_index_size']
embedding_dim = params['embedding_dimension']
categories = params['categories']

model = build_model(voc_size, embedding_dim, categories)

# Compile the model
model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])



hist = model.fit(x_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(x_val, y_val)
                )

# saving of model (should this be done differently ?)
model.save("models/phishing_model.h5")