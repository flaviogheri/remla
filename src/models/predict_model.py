# models/predict_model.py

import os
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model


# Load the configuration (if needed)
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


# Check if figures directories exist: 
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# Load the test data
x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")
y_test = y_test.reshape(-1, 1)


# Load the trained model
model = load_model("models/phishing_model.h5")

# Generate predictions
y_pred = model.predict(x_test, batch_size=1000)

# Convert predicted probabilities to binary labels
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

report = classification_report(y_test, y_pred_binary)

with open("reports/classification_report.txt", "w") as file:
    file.write("Classification Report:\n")
    file.write(report)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_binary)
with open("reports/confusion_matrix.txt", "w") as file:
    file.write("Confusion Matrix:\n")
    file.write(np.array_str(confusion_mat))

# Save accuracy to a file
accuracy = accuracy_score(y_test, y_pred_binary)
with open("reports/accuracy.txt", "w") as file:
    file.write(f"Accuracy: {accuracy}")

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.savefig("reports/figures/confusion_matrix_heatmap.png")
plt.close()