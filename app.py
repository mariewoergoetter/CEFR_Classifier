import streamlit as st
from text_feature_transformer import TextFeatureTransformer
from joblib import load

# Load the model with the best performance, in this case xgboost
model_path = 'xgboost_model.joblib'
model = load(model_path)

# Define the mapping from numeric, used in training, back to CEFR levels
cefr_mapping = {
    0: 'A1',
    1: 'A2',
    2: 'B1',
    3: 'B2',
    4: 'C1',
    5: 'C2'
}

def predict(text):
    prediction = model.predict([text])
    cefr_level = cefr_mapping[prediction[0]]
    return cefr_level

st.title('CEFR Level Classifier')

user_input = st.text_area("Enter the text you want to have assessed here:", "")

if st.button('Predict'):
    prediction = predict(user_input)
    st.write(f'Your level is probably: {prediction}')

