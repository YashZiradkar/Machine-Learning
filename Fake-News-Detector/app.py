import streamlit as st
from predictor import predict_news

st.title("📰 Fake News Detector")

article = st.text_area(
    "Paste a news article or headline"
)

if st.button("Analyze"):

    if article:

        result = predict_news(article)

        if result["label"] == 'LABEL_0':
            label = "Fake"
        else:
            label = "Real"

        st.subheader("Result")

        st.write(
            f"Prediction: {label}"
        )

        st.write(
            f"Confidence: {result['score']:.2%}"
        )
