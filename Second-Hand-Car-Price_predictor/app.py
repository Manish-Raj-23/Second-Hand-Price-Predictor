import joblib as jb
import pandas as pd
import streamlit as st
model = jb.load("Second-Hand-Car-Price_predictor/a_model.pkl")
encoded_name = jb.load("Second-Hand-Car-Price_predictor/encoded_name.pkl")
encoded_company = jb.load("Second-Hand-Car-Price_predictor/encoded_company.pkl")
encoded_fuel = jb.load("Second-Hand-Car-Price_predictor/encoded_fuel.pkl")
st.title("Second-Hand Car Price Prediction App")

# Create input fields for user inputs
name = st.text_input("Name or Model of the Car","Ex. Mahindra Jeep CL550")
company = st.text_input("Company of the Car", "Enter company name")
kms_driven = st.number_input("KMs Driven by the Car", min_value=0, step=1000, value=10000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel","LPG"])
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2024, step=1, value=2015)

# Button for prediction
if st.button("Predict"):
    # Format the inputs into the form expected by the model

    data = {
        'name':[name],
        'company':[company],
        'year':[year],
        'kms_driven':[kms_driven],
        'fuel_type':[fuel_type]
    }
    df = pd.DataFrame(data)
    df['name'] = encoded_name.transform(df['name'])
    df['company'] = encoded_company.transform(df['company'])
    df['fuel_type'] = encoded_fuel.transform(df['fuel_type'])
    pre = model.predict(df)
    st.success(f"The estimated car price is:  ₹{pre[0]:.2f}")
