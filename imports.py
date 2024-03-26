import os
from lxml import etree
import numpy as np
import pandas as pd
import re
import spacy
import textstat
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import xgboost as xgb

nlp = spacy.load("en_core_web_sm")

print("All necessary packages have been imported.")

