'''
This file is used to preprocess the data before training the model.
'''

import os

from sklearn.preprocessing import LabelEncoder
from keras.api.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer

import yaml
import numpy as np

# open config.yml
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_directory = os.path.dirname(path)
config_file = os.path.join(project_directory, "config.yml")

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

training_path = config['data_paths']['training_file']
test_path = config['data_paths']['test_file']
val_path = config['data_paths']['val_file']


# Load data using config (WIP)

# with open(training_path, "r") as train_file:
# train = [line.strip() for line in train_file.readlines()[1:]]

# with open(test_path, "r") as test_file:
# test = [line.strip() for line in test_file.readlines()]  # Assume no header

# with open(val_path, "r") as val_file:
# val = [line.strip() for line in val_file.readlines()]  # Assume no header


train_file_path = os.path.join(path, "data", "raw", "train_raw.txt")
test_file_path = os.path.join(path, "data", "raw", "test_raw.txt")
val_file_path = os.path.join(path, "data", "raw", "val_raw.txt")

# Load train data
with open(train_file_path, "r") as train_file:
    train = [line.strip() for line in train_file.readlines()[1:]]

# Load test data
with open(test_file_path, "r") as test_file:
    test = [line.strip() for line in test_file.readlines()]

# Load validation data
with open(val_file_path, "r") as val_file:
    val = [line.strip() for line in val_file.readlines()]

# Preprocess data
raw_x_train = [line.split("\t")[1] for line in train]
raw_y_train = [line.split("\t")[0] for line in train]

raw_x_test = [line.split("\t")[1] for line in test]
raw_y_test = [line.split("\t")[0] for line in test]

raw_x_val = [line.split("\t")[1] for line in val]
raw_y_val = [line.split("\t")[0] for line in val]
print("Succesfully preprocessed data")

# Tokenizing Dataset

tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
char_index = tokenizer.word_index
char_index_size = len(char_index)
config['params']['char_index_size'] = char_index_size

with open(config_file, "w") as file:
    yaml.dump(config, file, default_flow_style=False)

sequence_length = config['params']['sequence_length']

x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)
print("Succesfully tokenized dataset")

# Encoding Labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(raw_y_train)
y_val = encoder.transform(raw_y_val)
y_test = encoder.transform(raw_y_test)
print("Succesfully encoded labels")

project_directory = os.path.dirname(path)

# Create the directory if it doesn't exist
processed_data_directory = os.path.join(project_directory, "data", "processed")
os.makedirs(processed_data_directory, exist_ok=True)

# Save arrays
np.save(os.path.join(processed_data_directory, "x_train.npy"), x_train)
np.save(os.path.join(processed_data_directory, "x_val.npy"), x_val)
np.save(os.path.join(processed_data_directory, "x_test.npy"), x_test)

np.save(os.path.join(processed_data_directory, "y_train.npy"), y_train)
np.save(os.path.join(processed_data_directory, "y_val.npy"), y_val)
np.save(os.path.join(processed_data_directory, "y_test.npy"), y_test)
print("Successfully saved processed data")


# # Save Processed data
# def save_processed_data():
#     project_directory = os.path.dirname(path)

#     # Create the directory if it doesn't exist
#     processed_data_directory = os.path.join(project_directory, "data", "processed")
#     os.makedirs(processed_data_directory, exist_ok=True)

#     # Save arrays
#     np.save(os.path.join(processed_data_directory, "x_train.npy"), x_train)
#     np.save(os.path.join(processed_data_directory, "x_val.npy"), x_val)
#     np.save(os.path.join(processed_data_directory, "x_test.npy"), x_test)

#     np.save(os.path.join(processed_data_directory, "y_train.npy"), y_train)
#     np.save(os.path.join(processed_data_directory, "y_val.npy"), y_val)
#     np.save(os.path.join(processed_data_directory, "y_test.npy"), y_test)
#     print("Successfully saved processed data")

# def main():
#     save_processed_data()

# if __name__ == '__main__':
#     main()
