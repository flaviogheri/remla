# pylint: disable=all
# flake8: noqa

import pytest
import os
import tensorflow as tf


# ignore the test if in Github Actions as does not contain GPU's

@pytest.mark.skipif(os.getenv('CI') == 'true', reason="Skipping GPU test in CI environment")
def test_gpu_availability():
    """Testing for GPU availability"""
    assert tf.config.list_physical_devices('GPU'), "GPU isnt available on this device"

