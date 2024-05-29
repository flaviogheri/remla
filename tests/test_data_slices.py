
import pytest

import os
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.api.models import load_model
import pandas as pd
from remlapreprocesspy import preprocess

@pytest.fixture
def load_raw_test_data_and_model():

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load parameters and data
    project_directory = os.path.dirname(path) + "/model-training/"
    config_file = os.path.join(project_directory, "config.yml")
    
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Check if figures directories exist:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Path for raw test data
    test_path = config['data_paths']['test_file']

    # Load test data
    with open(test_path, "r") as test_file:
        test = [line.strip() for line in test_file.readlines()]
    
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    # Load the trained model
    model_path = config["processed_paths"]["model_path"]
    model = load_model(model_path)

    return raw_x_test,raw_y_test,model

def test_data_slices(load_raw_test_data_and_model):
    
    raw_x_test,raw_y_test,model = load_raw_test_data_and_model

    # Shorten lengths of datasets drastically to quicken testing by a lot
    raw_x_test = raw_x_test[0:50]
    raw_y_test = raw_y_test[0:50]

    slices = {
        "short_urls":   list(filter(lambda x: len(x) <= 30, raw_x_test)),
        "long_urls":    list(filter(lambda x: len(x) >  30, raw_x_test))
    }
    
    for slice_name, slice_data in slices.items():
        preprocessed_data = preprocess(slice_data)
        predictions = model.predict(preprocessed_data)
        
        assert predictions is not None, f"No predictions for slice: {slice_name}"
        assert len(predictions) == len(slice_data), f"Mismatch in predictions for slice: {slice_name}"

if __name__ == "__main__":
    
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load parameters and data
    project_directory = os.path.dirname(path) + "/model-training/"
    config_file = os.path.join(project_directory, "config.yml")
    
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Check if figures directories exist:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Path for raw test data
    test_path = config['data_paths']['test_file']

    # Load test data
    with open(test_path, "r") as test_file:
        test = [line.strip() for line in test_file.readlines()]
    
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    # Load the trained model
    model_path = config["processed_paths"]["model_path"]
    model = load_model(model_path)

    test_data_slices([raw_x_test[0:50],raw_y_test[0:50],model])