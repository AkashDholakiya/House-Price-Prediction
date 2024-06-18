import streamlit as st
from joblib import load
import numpy as np
import os

# Load the model
model_path = os.path.join(os.getcwd(), 'price_pred.joblib')

if os.path.isfile(model_path):
    model = load(model_path)
    st.write("Model loaded successfully.")
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

# Streamlit app
st.title("Boston Housing Price Prediction")

# Input fields for each feature with higher precision and placeholders
CRIM = st.number_input("Per capita crime rate by town (CRIM)", value=None, step=0.000001, format="%.6f", placeholder="Enter value...")
ZN = st.number_input("Proportion of residential land zoned for lots over 25,000 sq. ft. (ZN)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
INDUS = st.number_input("Proportion of non-retail business acres per town (INDUS)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
CHAS = st.selectbox("Charles River dummy variable (1 if tract bounds river; 0 otherwise) (CHAS)", [0, 1])
NOX = st.number_input("Nitric oxide concentration (NOX)", value=None, step=0.000001, format="%.6f", placeholder="Enter value...")
RM = st.number_input("Average number of rooms per dwelling (RM)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
AGE = st.number_input("Proportion of owner-occupied units built prior to 1940 (AGE)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
DIS = st.number_input("Weighted distances to five Boston employment centers (DIS)", value=None, step=0.000001, format="%.6f", placeholder="Enter value...")
RAD = st.number_input("Index of accessibility to radial highways (RAD)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
TAX = st.number_input("Full-value property tax rate per $10,000 (TAX)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
PTRATIO = st.number_input("Pupil-teacher ratio by town (PTRATIO)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
B = st.number_input("1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town (B)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")
LSTAT = st.number_input("Percentage of lower status of the population (LSTAT)", value=None, step=0.01, format="%.2f", placeholder="Enter value...")

# Collect inputs
input_features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_features)
    st.success(f"The predicted median value of owner-occupied homes (in $1000s) is: {prediction[0]:.2f}")

