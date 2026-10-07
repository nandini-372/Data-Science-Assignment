import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Absolute path to current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
data_path = os.path.join(BASE_DIR, "diabetes.csv")
df = pd.read_csv(data_path)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved successfully at:", model_path)
