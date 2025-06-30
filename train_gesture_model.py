import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load gesture data
data = pd.read_csv("gesture_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Use k=3 neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

accuracy = model.score(X, y)
print("Training accuracy:", accuracy)

# Save the model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as gesture_model.pkl")
