import pytest
import os
import yaml
import numpy as np
from keras.api.models import load_model
from sklearn.metrics import accuracy_score, f1_score, recall_score


@pytest.fixture
def load_test_data_and_model():

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load parameters and data
    project_directory = os.path.dirname(path) + "/model-training/"
    config_file = os.path.join(project_directory, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Check if figures directories exist:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Load the test data
    x_test = np.load(config["processed_paths"]["x_test"])
    y_test = np.load(config["processed_paths"]["y_test"])
    y_test = y_test.reshape(-1, 1)

    # Load the trained model
    model_path = config["processed_paths"]["model_path"]
    model = load_model(model_path)

    return x_test, y_test, model


def test_model_performance(load_test_data_and_model):
    # Generate predictions
    x_test, y_test, model = load_test_data_and_model
    y_pred = model.predict(x_test, batch_size=1000)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

    # report = classification_report(y_test, y_pred_binary)

    # # Calculate the confusion matrix
    # confusion_mat = confusion_matrix(y_test, y_pred_binary)

    # Save accuracy to a file
    accuracy = accuracy_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    assert accuracy > 0.6, "Model accuracy is below 0.9"
    assert recall > 0.6, "Model recall is below 0.9"
    assert f1 > 0.6, "Model f1 score is below 0.9"
