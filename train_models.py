from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import re
import spacy
import textstat
import xgboost as xgb

nlp = spacy.load("en_core_web_sm")

class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        # Fit label encoder on the target variable
        self.label_encoder.fit(y)
        return self

    def transform(self, X, y=None):
        features = np.array([self.preprocess_text(text) for text in X])
        return features

    def preprocess_text(self, text):
        text = self._simplify_punctuation(text)

        # Initialize spaCy document for linguistic features
        doc = nlp(text)

        # Compute readability scores and other features
        features = [
            textstat.flesch_reading_ease(text),
            textstat.smog_index(text),
            textstat.flesch_kincaid_grade(text),
            textstat.coleman_liau_index(text),
            textstat.automated_readability_index(text),
            textstat.dale_chall_readability_score(text),
            textstat.difficult_words(text),
            textstat.linsear_write_formula(text),
            textstat.gunning_fog(text),
            self.get_mean_parse_tree_depth(doc),
            self.get_mean_ents_per_sentence(doc)
        ]

        # Return features as a numpy array
        return np.array(features)

    @staticmethod
    def _simplify_punctuation(text):
        text = re.sub(r"[,:;()\-]", " ", text)
        text = re.sub(r"[\.!?]", ".", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def get_mean_parse_tree_depth(doc):
        if not doc or not list(doc.sents):
            return 0  # Return a default value if doc is empty or has no sentences
        depths = [TextFeatureTransformer._get_depth(sent.root) for sent in doc.sents if sent.root is not None]
        return np.mean(depths) if depths else 0

    @staticmethod
    def _get_depth(token, depth=1):
        depths = [TextFeatureTransformer._get_depth(child, depth + 1) for child in token.children]
        return max(depths, default=depth)

    @staticmethod
    def get_mean_ents_per_sentence(doc):
        num_sents = len(list(doc.sents))
        if num_sents == 0:
            return 0  # Avoid division by zero if there are no sentences
        num_ents = len(doc.ents)
        return num_ents / num_sents

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
        ('text_features', TextFeatureTransformer())
    ])),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', RandomForestClassifier())  # Placeholder classifier
])

X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['CEFR_Level'], test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVC': SVC(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

for name, classifier in classifiers.items():
    pipeline.set_params(classifier=classifier)
    print(f"Training {name}...")
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    pipeline.fit(X_train, y_train_encoded)

    y_pred = pipeline.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test_encoded, y_pred))
