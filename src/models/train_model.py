'''
This script trains the model using the data from the data/processed folder.
'''
import os
import time
import numpy as np
import yaml
from models import model_definition

path = os.path.dirname(os.path.abspath(__file__))

print(f"saving model in following directory: {os.path.dirname(path)}\\models\\phishing_model.keras")
# Load parameters and data
project_directory = os.path.dirname(os.path.dirname(path))
config_file = os.path.join(project_directory, "config.yml")
# print(config_file)
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

params = config['params']


# UKNOWN HERE : Do I load data using np.load or something else !?
x_train = np.load(path + "\\data\\processed\\x_train.npy")
y_train = np.load(path + "\\data\\processed\\y_train.npy")
x_val = np.load(path + "\\data\\processed\\x_val.npy")
y_val = np.load(path + "\\data\\processed\\y_val.npy")

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


# saving of model (should this be done differently ?)
model.save(os.path.dirname(path) + "\\models\\phishing_model.keras")

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Training completed in {minutes}:{seconds}")
