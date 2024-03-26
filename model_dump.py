from joblib import dump

filename = f'{best_model_name.replace(" ", "_").lower()}_best_model.joblib'
dump(best_model, filename)
print(f'Model saved: {filename}')

