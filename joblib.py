from joblib import dump

# Assuming your last trained pipeline is in a variable named 'pipeline'
# Define the path and filename for saving the model
model_filename = '/Users/mlwfee/Desktop/CEFR_Classifier/xgboost_model.joblib'

# Save the model
dump(pipeline, model_filename)

print(f"Model saved to {model_filename}")
