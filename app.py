import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

model = joblib.load("data/model.pkl")
features = joblib.load("data/features.pkl")
analyzer = SentimentIntensityAnalyzer()

st.set_page_config(
    page_title="Amazon Return Risk Predictor",
    page_icon="🔍",
    layout="centered"
)

st.title("Amazon Return Risk Predictor")
st.markdown("""
**The insight:** Star ratings don't predict returns. 
A product can have 4.6 stars and still get returned 35% of the time.
This tool detects the gap between what customers *write* and what they *rate*.
""")

st.divider()

st.subheader("Paste an Amazon review")

review_text = st.text_area(
    "Review text",
    placeholder="e.g. The battery dies after 3 hours. Not what I expected for the price...",
    height=150
)

rating = st.slider("Star rating given", 1.0, 5.0, 4.0, step=0.5)

if st.button("Predict Return Risk", type="primary"):
    if not review_text.strip():
        st.warning("Please paste a review first.")
    else:
        sentiment = analyzer.polarity_scores(review_text)['compound']
        rating_normalized = (rating - 3) / 2
        mismatch_score = rating_normalized - sentiment
        review_length = len(review_text.split())
        has_exclamation = int('!' in review_text)

        input_data = pd.DataFrame([{
            'mismatch_score': mismatch_score,
            'sentiment_score': sentiment,
            'review_length': review_length,
            'has_exclamation': has_exclamation,
            'rating': rating
        }])

        prob = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Return Risk Score", f"{prob:.1%}")
        with col2:
            st.metric("Sentiment Score", f"{sentiment:.3f}")
        with col3:
            st.metric("Mismatch Score", f"{mismatch_score:.3f}")

        if prob >= 0.7:
            st.error("🚨 High return risk — review sentiment significantly mismatches star rating")
        elif prob >= 0.4:
            st.warning("⚠️ Moderate return risk — some signals of mismatch detected")
        else:
            st.success("✅ Low return risk — review text aligns with star rating")

        st.divider()
        st.subheader("Why this prediction?")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        fig, ax = plt.subplots(figsize=(8, 3))
        shap_vals = shap_values[0]
        colors = ['#D85A30' if v > 0 else '#534AB7' for v in shap_vals]
        ax.barh(features, shap_vals, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title("Feature contribution to this prediction", fontsize=12)
        ax.set_xlabel("SHAP value (red = increases risk, purple = decreases risk)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.caption(
            "Model: XGBoost trained on 34k Amazon Electronics reviews | "
            "Proxy label: high-rated reviews with low sentiment + zero helpful votes | "
            "AUC-ROC: 1.000 | "
            "[View on GitHub](https://github.com/rimiify)"
        )