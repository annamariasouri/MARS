import joblib

# Load the existing model using the current environment
model = joblib.load("final_rf_chl_model_2015_2023.pkl")

# Re-save the model using protocol 4 to improve compatibility
joblib.dump(model, "new_updated_model.pkl", protocol=4)
print("Model has been re-saved as new_updated_model.pkl using protocol 4.")