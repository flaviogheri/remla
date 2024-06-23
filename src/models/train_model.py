'''
This script trains the model using the data generated in the previous step.
The model is saved in the models directory.
'''

import os
import numpy as np
import yaml
# import src.models.model_definition as model_definition
import model_definition
import os
import time

if __name__ == "__main__":

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_directory = os.path.dirname(path)
    model_directory = os.path.join(project_directory, "models", "phishing_model.keras")
    print(f"saving model in the following directory: {model_directory}")

    # Load parameters and data
    config_file = os.path.join(project_directory, "config.yml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    params = config['params']

    data_directory = os.path.join(project_directory, "data", "processed")

    # Load data
    x_train = np.load(config["processed_paths"]["x_train"])
    y_train = np.load(config["processed_paths"]["y_train"])
    x_val = np.load(config["processed_paths"]["x_val"])
    y_val = np.load(config["processed_paths"]["y_val"])

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
    model.save(config["processed_paths"]["model_path"])

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training completed in {minutes}:{seconds}")

def return_model_directly(config_file_path,slice_amount = 0.01):
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_directory = os.path.dirname(path)
    
    # Load parameters and data
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    params = config['params']

    data_directory = os.path.join(project_directory, "data", "processed")

    # Load data
    x_train = np.load(config["processed_paths"]["x_train"])
    y_train = np.load(config["processed_paths"]["y_train"])
    x_val = np.load(config["processed_paths"]["x_val"])
    y_val = np.load(config["processed_paths"]["y_val"])

    # Shorten data to quicken training for testing purposes
    x_train = x_train[0:int(len(x_train)*slice_amount)]
    y_train = y_train[0:int(len(y_train)*slice_amount)]
    x_val   = x_val[0:int(len(x_val)*slice_amount)]
    y_val   = y_val[0:int(len(y_val)*slice_amount)]

    # Build the model
    voc_size = params['char_index_size']
    embedding_dim = params['embedding_dimension']
    categories = params['categories']

    model = model_definition.build_model()

    # Compile the model
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
                    batch_size=params['batch_train'],
                    epochs=params['epoch'],
                    shuffle=True,
                    validation_data=(x_val, y_val)
                    )
    return model