import spacy
import textstat
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import re

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Fit method is used when the model requires fitting to the data,
        # but this transformer doesn't, so we just return self
        return self

    def transform(self, X, y=None):
        # Transform the text data to features
        features = np.array([self.preprocess_text(text) for text in X])
        return features

    @staticmethod
    def preprocess_text(text):
        # Simplify the text to help with the readability scores
        text = TextFeatureTransformer._simplify_punctuation(text)

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
            TextFeatureTransformer.get_mean_parse_tree_depth(doc),
            TextFeatureTransformer.get_mean_ents_per_sentence(doc)
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


