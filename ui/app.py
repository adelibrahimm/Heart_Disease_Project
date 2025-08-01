import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


model = joblib.load("models/final_model.pkl") 
data = pd.read_csv("data/selected_features.csv")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Prediction App")


st.sidebar.header("Patient Information")

def user_input():
    input_data = {}
    for col in data.drop("target", axis=1).columns:
        val = st.sidebar.slider(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
        input_data[col] = val
    return pd.DataFrame([input_data])

input_df = user_input()


if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result")
    st.write(f"Prediction: **{'Heart Disease' if prediction == 1 else 'No Heart Disease'}**")
    st.write(f"Probability: **{prob:.2f}**")


if st.checkbox("Show PCA Visualization"):
    pca_df = pd.read_csv("data/pca_transformed.csv")
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='target', palette='coolwarm', ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Sample Dataset"):
    st.dataframe(data.sample(10))
