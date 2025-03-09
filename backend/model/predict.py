import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def predict_recipe(cuisine, diet, dishType):
    # Encode input values
    cuisine = label_encoders["cuisine"].transform([cuisine])[0]
    diet = label_encoders["diet"].transform([diet])[0]
    dishType = label_encoders["dishType"].transform([dishType])[0]

    # Make prediction
    input_data = np.array([[cuisine, diet, dishType]])
    prediction = model.predict(input_data)

    return prediction[0]
