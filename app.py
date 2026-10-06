import pickle
from pathlib import Path

import streamlit as st

from sms_preprocessing import ensure_nltk_resources, transform_text


PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    ensure_nltk_resources()
    with (PROJECT_DIR / "vectorizer.pkl").open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with (PROJECT_DIR / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model


st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Enter a message to classify.")
    else:
        tfidf, model = load_artifacts()
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("Spam" if result == 1 else "Not Spam")