import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("./newfood.csv")

# Feature selection
features = ["calories", "Fat/g", "Protein/g", "Sodium/mg", "Sugar/g"]
X = df[features]
y = df["healthScore"]  # Target variable (higher is better)

# Train model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X, y)

# Save model
with open("model/decision_tree.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete!")
