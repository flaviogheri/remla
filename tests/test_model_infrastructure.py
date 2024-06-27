# pylint: disable=all
# flake8: noqa

import pytest
import os
import tensorflow as tf


def test_gpu_availability():
    """Testing for GPU availability"""
    assert tf.config.list_physical_devices('GPU'), "GPU isnt available on this device"

