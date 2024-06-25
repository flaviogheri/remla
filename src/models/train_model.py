'''
This script trains the model using the data generated in the previous step.
The model is saved in the models directory.
'''

import os
import time
import numpy as np
import yaml
import tensorflow as tf
from src.models import model_definition  # pylint: disable=import-error


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_directory = os.path.dirname(path)
    model_directory = os.path.join(project_directory, "models", "phishing_model.keras")
    print(f"saving model in the following directory: {model_directory}")

    # Load parameters and data
    config_file_path = os.path.join(project_directory, "config.yml")
    with open(config_file_path, "r", encoding="utf-8") as file:
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

    hist = model.fit(
                    x_train,
                    y_train,
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


def return_model_directly(config_path, slice_amount=0.01):
    '''
    This function trains a model using loaded data and returns it directly
    '''

    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

    # Load parameters and data
    with open(config_path, "r", encoding="utf-8") as config_file:
        configuration = yaml.safe_load(config_file)

    parameters = configuration['params']

    # Load data
    loaded_x_train = np.load(configuration["processed_paths"]["x_train"])
    loaded_y_train = np.load(configuration["processed_paths"]["y_train"])
    loaded_x_val = np.load(configuration["processed_paths"]["x_val"])
    loaded_y_val = np.load(configuration["processed_paths"]["y_val"])

    # Shorten data to quicken training for testing purposes
    loaded_x_train = loaded_x_train[0:int(len(loaded_x_train)*slice_amount)]
    loaded_y_train = loaded_y_train[0:int(len(loaded_y_train)*slice_amount)]
    loaded_x_val = loaded_x_val[0:int(len(loaded_x_val)*slice_amount)]
    loaded_y_val = loaded_y_val[0:int(len(loaded_y_val)*slice_amount)]

    built_model = model_definition.build_model()

    # Compile the model
    built_model.compile(
                       loss=parameters['loss_function'],
                       optimizer=parameters['optimizer'],
                       metrics=['accuracy']
                       )

    built_model.fit(
                   loaded_x_train,
                   loaded_y_train,
                   batch_size=parameters['batch_train'],
                   epochs=parameters['epoch'],
                   shuffle=True,
                   validation_data=(loaded_x_val, loaded_y_val)
                   )

    return built_model
