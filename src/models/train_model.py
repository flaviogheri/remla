
import numpy as np
import yaml
from models import model_definition
import os
import time

print(f"saving model in the following directory: {os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))}\\models\\phishing_model.keras")
# Load parameters and data
project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_file = os.path.join(project_directory, "config.yml")
#print(config_file)
with open(config_file, "r") as file: 
    config = yaml.safe_load(file)

params = config['params']


# UKNOWN HERE : Do I load data using np.load or something else !?
x_train = np.load(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "\\data\\processed\\x_train.npy")
y_train = np.load(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "\\data\\processed\\y_train.npy")
x_val = np.load(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "\\data\\processed\\x_val.npy")
y_val = np.load(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + "\\data\\processed\\y_val.npy")

# Build the model
voc_size = params['char_index_size']
embedding_dim = params['embedding_dimension']
categories = params['categories']

model = model_definition.build_model(embedding_dim, categories)

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
model.save(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))) + "\\models\\phishing_model.keras")

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Training completed in {minutes}:{seconds}")
