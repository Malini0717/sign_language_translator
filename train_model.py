# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load the data
df = pd.read_csv("gesture_data.csv")

# Step 2: Split into features and labels
X = df.iloc[:, :-1]  # First 63 columns (x, y, z of landmarks)
y = df.iloc[:, -1]   # Last column (label like A, B, etc.)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Save the trained model
joblib.dump(model, "gesture_model.pkl")

# Step 6: Print model accuracy (optional)
accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.2f}")
