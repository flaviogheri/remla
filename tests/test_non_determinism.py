import numpy as np
from src.models import train_model
import remlapreprocesspy
import tensorflow as tf


# Test non-determininsm when training using settings within testing_config.yml
def test_non_determininsm_robustness():
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

    model_1 = train_model.return_model_directly(config_path="testing_config.yml")

    tf.keras.utils.set_random_seed(1)
    model_2 = train_model.return_model_directly(config_path="testing_config.yml")

    processed = remlapreprocesspy.preprocess("http://www.testurl.org")

    prediction_1 = model_1.predict(processed)
    prediction_2 = model_2.predict(processed)

    assert np.allclose(prediction_1, prediction_2, atol=1e-6), "Different predictions but same seed"


if __name__ == "__main__":
    test_non_determininsm_robustness()
