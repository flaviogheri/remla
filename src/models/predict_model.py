# models/predict_model.py


# UNSURE ABOUT THIS SECTION :: DO I NOT WANT TO JUST HAVE THIS OUTPUT WHATEVER
# INPUT ONE GIVES INSTEAD OF DOING IT FOR THE TEST DATA ONLY ?

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import yaml


# Load the configuration (if needed)
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Load the test data
x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")
y_test = y_test.reshape(-1, 1)


# Load the trained model
model = load_model("models/phishing_model.h5")


### Part below : Believed to be unnecessary ...



# Generate predictions
y_pred = model.predict(x_test, batch_size=1000)

# Convert predicted probabilities to binary labels
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

# Calculate the classification report
report = classification_report(y_test, y_pred_binary)
print("Classification Report:")
print(report)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:\n", confusion_mat)
print("Accuracy:", accuracy_score(y_test, y_pred_binary))

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()