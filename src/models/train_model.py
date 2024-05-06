'''
This script trains the model using the data generated in the previous step.
The model is saved in the models directory.
'''

import os
import time

import numpy as np
import yaml
from models import model_definition

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_directory = os.path.dirname(path)
model_directory = os.path.join(project_directory, "models", "phishing_model.keras")
print(f"saving model in the following directory: {model_directory}")

# Load parameters and data
config_file = os.path.join(project_directory, "config.yml")
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

params = config['params']

# Load data
data_directory = os.path.join(project_directory, "data", "processed")
x_train = np.load(os.path.join(data_directory, "x_train.npy"))
y_train = np.load(os.path.join(data_directory, "y_train.npy"))
x_val = np.load(os.path.join(data_directory, "x_val.npy"))
y_val = np.load(os.path.join(data_directory, "y_val.npy"))

# Build the model
voc_size = params['char_index_size']
embedding_dim = params['embedding_dimension']
categories = params['categories']

model = model_definition.build_model()

# Compile the model
model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

start_time = time.time()

hist = model.fit(x_train, y_train,
                 batch_size=params['batch_train'],
                 epochs=params['epoch'],
                 shuffle=True,
                 validation_data=(x_val, y_val)
                 )

# Save the model
model.save(model_directory)

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Training completed in {minutes}:{seconds}")
