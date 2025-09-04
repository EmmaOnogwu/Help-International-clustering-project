import streamlit as st
import pandas as pd
import joblib

# App Title
st.title("Clustering Project")

# User Inputs
health = st.number_input("Enter Total Health Amount", min_value=0.0, step=1.0)
trade = st.number_input("Enter Total Trade Amount", min_value=0.0, step=1.0)
finance = st.number_input("Enter Total Finance Amount", min_value=0.0, step=1.0)

# Create DataFrame from inputs
input_data = pd.DataFrame({
    "Health": [health],
    "Trade": [trade],
    "Finance": [finance]
})

# Load pre-trained model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Button to trigger prediction
if st.button("Predict Cluster"):
    # Scale input data using the pre-fitted scaler
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display Result
    st.subheader("Cluster Prediction")
    choice = {0: "Not a priority", 
            1: "Requires foreign aid",
            2: "Do NOT requires foreign aid"
            }



    st.success(f"✅ The input data belongs to **Cluster {prediction[0]}**: {choice.get(prediction[0])}")
