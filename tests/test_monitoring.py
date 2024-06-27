# pylint: disable=all
# flake8: noqa


import pytest
import os
import yaml
import numpy as np
import time
import remlapreprocesspy


import tensorflow as tf
from keras.api.models import load_model

from src.models import model_definition
from src.models import train_model


import psutil

def ram_usage():
    process = psutil.Process()
    memory = process.memory_info()

    return memory


def test_model_training_time():
    max_time = 100 # max allowed training time
    tf.keras.utils.set_random_seed(1)

    # Training speed test

    time_init = time.time()

    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

    # training the model
    _ = train_model.return_model_directly(config_path="testing_config.yml")

    time_end = time.time()

    # final training time
    training_time = time_end - time_init

    print("training time: ", training_time)
    assert training_time <= max_time, f"Training took too long, limit : {max_time}s, result: {training_time:.2f}s"


def test_ram_usage():
    # compairing the ram usage between existing and new model

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load parameters and data
    project_directory = os.path.dirname(path) + "/model-training/"
    config_file = os.path.join(project_directory, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    x_test = np.load(config["processed_paths"]["x_test"])
    y_test = np.load(config["processed_paths"]["y_test"])
    y_test = y_test.reshape(-1, 1)

    model_path = config["processed_paths"]["model_path"]
    _ = load_model(model_path)

    # find ram of existing model
    original_train_ram = ram_usage() / (1024 **2)

    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

    _ = train_model.return_model_directly(config_path="testing_config.yml")
    
    
    recent_train_ram = ram_usage() / (1024 **2)

    assert (recent_train_ram - original_train_ram) < 100, "Memory increase between models is too large"
    


if __name__ == "__main__":
    test_model_training_time()
    test_ram_usage()
