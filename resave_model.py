import joblib

# Load the existing model using the current environment
model = joblib.load("final_rf_chl_model_2015_2023.pkl")

# Re-save the model using protocol 5 as a new file named 'new_updated_model.pkl'
joblib.dump(model, "new_updated_model.pkl", protocol=5)
print("Model has been re-saved as new_updated_model.pkl.")