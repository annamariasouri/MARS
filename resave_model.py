import joblib

Load the existing model using the current environment
model = joblib.load("final_rf_chl_model_2015_2023.pkl")

Re-save the model using protocol 5
joblib.dump(model, "final_rf_chl_model_2015_2023.pkl", protocol=5) print("Model has been re-saved.")