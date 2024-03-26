from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

#Defininghyperparameter search space for each model
param_grids = {
    'Random Forest': {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(10, 50),
        'classifier__min_samples_split': randint(2, 11),
        'classifier__min_samples_leaf': randint(1, 11)
    },
    'Logistic Regression': {
        'classifier__C': uniform(0.01, 10),
        'classifier__penalty': ['l2']
    },
    'SVC': {
        'classifier__C': uniform(0.1, 10),
        'classifier__kernel': ['linear', 'rbf']
    },
    'XGBoost': {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.3)
    }
}

#Dictionary to store the best estimator for each model
best_estimators = {}

for name, param_grid in param_grids.items():
    print(f"Optimizing {name}...")
    
    if name == 'Random Forest':
        classifier = RandomForestClassifier()
    elif name == 'Logistic Regression':
        classifier = LogisticRegression(max_iter=1000)
    elif name == 'SVC':
        classifier = SVC()
    elif name == 'XGBoost':
        classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    pipeline.set_params(classifier=classifier)

    search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=30, cv=3, scoring='accuracy', random_state=42, verbose=1)

    search.fit(X_train, y_train_encoded)

    best_estimators[name] = search.best_estimator_
    
    print(f"Best score for {name}: {search.best_score_}")
    print(f"Best parameters for {name}: {search.best_params_}\n")

# Final testing
print("Final Testing Results:")
for name, estimator in best_estimators.items():
    y_pred = estimator.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test_encoded, y_pred))

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

