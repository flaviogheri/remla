import os
import yaml
import pytest

@pytest.fixture(scope="module")
def load_data():
    # Load the configuration file
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_directory = os.path.dirname(path)
    config_file = os.path.join(project_directory, "config.yml")

    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_path = config['data_paths']['training_file']
    test_path = config['data_paths']['test_file']
    val_path = config['data_paths']['val_file']

    # Load train, test, and validation data
    def _load_data(path):
        with open(path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file.readlines()[1:]]

    train = _load_data(training_path)
    test = _load_data(test_path)
    val = _load_data(val_path)

    return train, test, val

def test_preprocess_data(load_data):
    train, test, val = load_data

    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    raw_x_val = [line.split("\t")[1] for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    assert len(raw_x_train) == len(raw_y_train), "Mismatch between X and Y in training data"
    assert len(raw_x_test) == len(raw_y_test), "Mismatch between X and Y in test data"
    assert len(raw_x_val) == len(raw_y_val), "Mismatch between X and Y in validation data"

if __name__ == "__main__":
    pytest.main()
