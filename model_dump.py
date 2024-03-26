from joblib import dump

model_filename = 'xgboost_model.joblib'

dump(pipeline, model_filename)

print(f"Model saved to {model_filename}")
