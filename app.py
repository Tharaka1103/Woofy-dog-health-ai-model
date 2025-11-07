from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes
MY_SECRET_API_KEY = "sk_woofy_a9z8y7x6w5hgy4f374tfn83wgt86ygh3y4ntc34t3cti3c7r34ctr78"
# --- Load the trained model, LabelEncoder, and feature columns ---
model_pipeline = None
label_encoder = None
feature_columns = None

try:
    model_pipeline = joblib.load('dog_health_model.joblib')
    label_encoder = joblib.load('health_label_encoder.joblib')
    feature_columns = joblib.load('feature_columns.joblib') # Load the feature columns list
    
    print("Model pipeline, LabelEncoder, and feature columns loaded successfully!")
    print(f"Expected Features: {feature_columns}")

except FileNotFoundError as e:
    print(f"Error: One or more required files not found: {e}")
    print("Make sure 'dog_health_model.joblib', 'health_label_encoder.joblib', and 'feature_columns.joblib' are in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred loading resources: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    # --- ADD THIS SECURITY CHECK ---
    auth_header = request.headers.get('X-API-Key')
    if auth_header != MY_SECRET_API_KEY:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)
    
    # Check if all required features are present in the input data
    missing_features = [col for col in feature_columns if col not in data]
    if missing_features:
        return jsonify({
            "status": "error",
            "message": f"Missing required features: {', '.join(missing_features)}"
        }), 400

    # Convert input data to DataFrame, ensuring correct column order and only required columns
    try:
        # Create a DataFrame from the input data, but only for the expected feature_columns
        # This handles cases where extra data might be sent, and ensures correct order.
        input_values = {col: [data.get(col)] for col in feature_columns}
        input_df = pd.DataFrame(input_values, columns=feature_columns)
        
        # Make prediction (returns an array, so take the first element)
        prediction_numeric = model_pipeline.predict(input_df)[0]
        
        # Convert numeric prediction back to original label using the loaded LabelEncoder
        prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]
        
        friendly_prediction = ""
        if prediction_label == 'Yes':
            friendly_prediction = 'Healthy'
        elif prediction_label == 'No':
            friendly_prediction = 'Not Healthy'
        else:
            friendly_prediction = prediction_label # Fallback in case of unexpected values
        
        # Return the new friendly_prediction
        return jsonify({"status": "success", "prediction": friendly_prediction}), 200    

    except Exception as e:
        print(f"Prediction error: {e}")
        # Log the full traceback for debugging purposes
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible from other devices on your local network,
    # useful for testing with a phone browser. For local dev, 127.0.0.1 is fine too.
    app.run(host='0.0.0.0', port=5000, debug=True)