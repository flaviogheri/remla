# import pytest 
# import os
# from src.models import predict_model
# import time 


# # test machine learning model infrastructure

# @pytest.fixture
# def load_test_data_and_model():

#     path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     # Load parameters and data
#     project_directory = os.path.dirname(path) + "/model-training/"
#     config_file = os.path.join(project_directory, "config.yml")
    
#     with open(config_file, "r") as file:
#         config = yaml.safe_load(file)

#     # Check if figures directories exist:
#     os.makedirs("reports", exist_ok=True)
#     os.makedirs("reports/figures", exist_ok=True)

#     # Load the test data
#     x_test = np.load(config["processed_paths"]["x_test"])
#     y_test = np.load(config["processed_paths"]["y_test"])
#     y_test = y_test.reshape(-1, 1)

#     # Load the trained model
#     model_path = config["processed_paths"]["model_path"]
#     model = load_model(model_path)

#     return x_test,y_test,model

# # compare model with previous model ?

# def test_model_rollback(load_test_data_and_models)
#     x_test, y_test, current_model, prev_model = load_test_data_and_models

#     # Measure the time it takes to switch from the current model to the previous model
#     start_time = time.time()
#     model = prev_model  # switch to the previous model
#     end_time = time.time()
#     rollback_time = end_time - start_time
#     print(f"Rollback time: {rollback_time} seconds")

#     # Verify that the previous model is working correctly
#     y_pred = model.predict(x_test, batch_size=1000)
#     y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
#     report = classification_report(y_test, y_pred_binary)
#     print(report)