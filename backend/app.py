import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import traceback

# # Load dataset
# df = pd.read_csv("newfood.csv")

# # Features
# features = [
#     "calories", "Fat/g", "Saturated Fat/g", "Carbohydrates/g", "Sugar/g",
#     "Cholesterol/mg", "Sodium/mg", "Protein/g"
# ]
# X = df[features]

# # Suitability conditions (BP, diabetes, heart disease)
# df["suitability"] = (
#     ((df["Sodium/mg"] < 400)) &   # Low sodium for BP
#     ((df["Sugar/g"] < 10)) &      # Low sugar for diabetes
#     ((df["Cholesterol/mg"] < 50)) # Low cholesterol for heart disease
# ).astype(int)

# y = df["suitability"]

# # Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

# # ✅ Train a Random Forest Classifier for better generalization
# model = RandomForestClassifier(
#     n_estimators=100, max_depth=7, min_samples_split=5, min_samples_leaf=2,
#     class_weight="balanced", random_state=2
# )
# model.fit(X_train, y_train)

# # Save model
# with open("recipe_recommender.pkl", "wb") as f:
#     pickle.dump(model, f)

# print("✅ Model trained successfully!")

# # Cross-validation
# scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
# print(f"📊 Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Load dataset
df = pd.read_csv("newfood.csv")

# Features
features = [
    "calories", "Fat/g", "Saturated Fat/g", "Carbohydrates/g", "Sugar/g",
    "Cholesterol/mg", "Sodium/mg", "Protein/g"
]
X = df[features]

# Suitability conditions (BP, diabetes, heart disease)
df["suitability"] = (
    ((df["Sodium/mg"] < 400)) &   # Low sodium for BP
    ((df["Sugar/g"] < 10)) &      # Low sugar for diabetes
    ((df["Cholesterol/mg"] < 50)) # Low cholesterol for heart disease
).astype(int)

y = df["suitability"]

# Split dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=2, class_weight="balanced", C=0.1)

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Test accuracy: {test_accuracy:.4f}")

# Save the trained model using joblib
joblib.dump(model, "recipe_recommender.pkl")

# Save the features and labels for reference
joblib.dump(features, "features.pkl")
joblib.dump(y, "labels.pkl")


# ------------------- FLASK BACKEND -------------------

app = Flask(__name__)
CORS(app)

# Load the trained model using joblib
try:
    model = joblib.load("recipe_recommender.pkl")  # Use joblib.load instead of pickle.load
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", str(e))

# Expected columns
REQUIRED_COLUMNS = ["calories", "Fat/g", "Saturated Fat/g", "Carbohydrates/g",
                    "Sugar/g", "Cholesterol/mg", "Sodium/mg", "Protein/g"]

# Function to extract required nutrients
def extract_nutrients(nutrients_list):
    nutrient_dict = {
        'Calories': 'calories',
        'Fat': 'Fat/g',
        'Saturated Fat': 'Saturated Fat/g',
        'Carbohydrates': 'Carbohydrates/g',
        'Sugar': 'Sugar/g',
        'Cholesterol': 'Cholesterol/mg',
        'Sodium': 'Sodium/mg',
        'Protein': 'Protein/g'
    }
    
    extracted = {col: None for col in REQUIRED_COLUMNS}  # Initialize all required columns
    
    for nutrient in nutrients_list:
        name = nutrient.get('name')
        amount = nutrient.get('amount')
        if name in nutrient_dict:
            extracted[nutrient_dict[name]] = amount
    
    return extracted

# Function to check suitability based on user profile
def is_suitable(recipe, user_profile):
    diabetes = user_profile.get("diabetes", False)
    high_bp = user_profile.get("high_bp", False)
    heart_disease = user_profile.get("heart_disease", False)

    suitability_score = 0
    if diabetes and recipe.get("Sugar/g", 0) >= 10:
        suitability_score += 1  # Penalize if sugar is too high for diabetes
    if high_bp and recipe.get("Sodium/mg", 0) >= 400:
        suitability_score += 1  # Penalize if sodium is too high for high BP
    if heart_disease and recipe.get("Cholesterol/mg", 0) >= 50:
        suitability_score += 1  # Penalize if cholesterol is too high for heart disease
    
    # Additional flexibility: allow for a small deviation for user profile compatibility
    deviation_tolerance = user_profile.get("deviation_tolerance", 0.1)
    if suitability_score / 3 <= deviation_tolerance:
        return True
    return suitability_score == 0  # If no penalty, the recipe is suitable

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        print("🔹 Received JSON:", data)

        if not data or "recipes" not in data or "userProfile" not in data:
            return jsonify({"error": "Missing 'recipes' or 'userProfile' in request"}), 400

        recipes = data["recipes"]
        user_profile = data["userProfile"]

        if not recipes:
            return jsonify({"error": "No recipes provided"}), 400

        processed_recipes = []
        for recipe in recipes:
            recipe_data = {
                "id": recipe.get("id"),
                "title": recipe.get("title"),
                "image": recipe.get("image"),
                "usedIngredientCount": recipe.get("usedIngredientCount", 0),
                "missedIngredientCount": recipe.get("missedIngredientCount", 0),
            }

            # Extract nutrients
            if "nutrition" in recipe and "nutrients" in recipe["nutrition"]:
                nutrition_info = extract_nutrients(recipe["nutrition"]["nutrients"])
                recipe_data.update(nutrition_info)
            processed_recipes.append(recipe_data)

        df_recipes = pd.DataFrame(processed_recipes)
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_recipes.columns]
        if missing_cols:
            print("❌ Missing columns:", missing_cols)
            return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

        df_recipes.fillna(0, inplace=True)

        # Predict suitability
        predictions = model.predict(df_recipes[REQUIRED_COLUMNS])

        recommended_recipes = []
        for recipe, suitable in zip(recipes, predictions):
            if suitable == 1 and is_suitable(recipe, user_profile):
                recommended_recipes.append(recipe)

        print(f"🔹 Recommended Recipes: {recommended_recipes}")
        return jsonify({"message": "Success", "recipes": recommended_recipes[:3]})

    except Exception as e:
        print("❌ Exception:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
