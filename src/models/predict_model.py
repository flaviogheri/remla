# models/predict_model.py

import os
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model


# Load parameters and data
project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_file = os.path.join(project_directory, "config.yml")
#print(config_file)
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
model = load_model(config["processed_paths"]["model_path"])

# Generate predictions
y_pred = model.predict(x_test, batch_size=1000)

# Convert predicted probabilities to binary labels
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

report = classification_report(y_test, y_pred_binary)

with open(config["report_paths"]["classification_report"], "w") as file:
    file.write("Classification Report:\n")
    file.write(report)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_binary)
with open(config["report_paths"]["confusion_matrix"], "w") as file:
    file.write("Confusion Matrix:\n")
    file.write(np.array_str(confusion_mat))

# Save accuracy to a file
accuracy = accuracy_score(y_test, y_pred_binary)
with open(config["report_paths"]["accuracy_score"], "w") as file:
    file.write(f"Accuracy: {accuracy}")

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.savefig(config["report_paths"]["heatmap"])
plt.close()
