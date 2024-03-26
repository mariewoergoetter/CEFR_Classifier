import streamlit as st
from text_feature_transformer import TextFeatureTransformer
from joblib import load

# Load your trained model
model_path = 'xgboost_model.joblib'
model = load(model_path)

# Define the mapping from numeric predictions back to CEFR levels
# This should match the encoding used during training
cefr_mapping = {
    0: 'A1',
    1: 'A2',
    2: 'B1',
    3: 'B2',
    4: 'C1',
    5: 'C2'
}

def predict(text):
    # This function makes a prediction based on the input text.
    prediction = model.predict([text])
    # Use the mapping to convert numeric prediction back to CEFR level
    cefr_level = cefr_mapping[prediction[0]]
    return cefr_level

# Streamlit app
st.title('CEFR Level Classifier')

# User text input
user_input = st.text_area("Enter the text you want to have assessed here:", "")

if st.button('Predict'):
    # Display the prediction
    prediction = predict(user_input)
    st.write(f'Your level is probably: {prediction}')

