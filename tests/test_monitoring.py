import time 
import os
import subprocess
import pytest


# from keras.api.models import load_model
# import psutil



MAX_TRAINING_TIME = 60  # [s]

def test_training_speed():
    
    start_time = time.time()

    # Train the model
    # model = train_model.py

    ### cant get env to run the python files, using a temporary file instead:
    subprocess.run(["python", "temp_test.py"])

    end_time = time.time()

    training_time = end_time - start_time
    assert training_time < MAX_TRAINING_TIME, "Training time has regressed"
