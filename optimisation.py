
param_grids = {
    'Random Forest': {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
    },
    'Logistic Regression': {
        'classifier': [LogisticRegression()],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
    },
    'SVC': {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf'],
    },
    'XGBoost': {
        'classifier': [xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6, 9],
    }
}


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

best_models = {}
best_scores = {}

for model_name, params in param_grids.items():
    print(f"Running GridSearchCV for {model_name}.")
    search = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='accuracy', verbose=2)
    search.fit(X_train, y_train_encoded)
    
    best_models[model_name] = search.best_estimator_
    best_scores[model_name] = search.best_score_
    
    print(f"Best Score for {model_name}: {search.best_score_}")
    print(f"Best Parameters for {model_name}: {search.best_params_}\n")

best_model_name = max(best_scores, key=best_scores.get)
print(f"Best model based on CV score: {best_model_name}")


best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test)
print(f"Classification Report for {best_model_name}:")
print(classification_report(y_test_encoded, y_pred))

