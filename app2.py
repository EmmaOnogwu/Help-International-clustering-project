import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Clustering Project", page_icon="🧩", layout="wide")

# Title
st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">🧩 Clustering Project</h1>
    <p style="text-align:center; color:gray;">Predicting which cluster your data belongs to</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Sidebar for inputs
st.sidebar.header("⚙️ Input Parameters")
health = st.sidebar.number_input("Enter Total Health Amount", min_value=0.0, step=1.0, help="Budget allocated to Health")
trade = st.sidebar.number_input("Enter Total Trade Amount", min_value=0.0, step=1.0, help="Budget allocated to Trade")
finance = st.sidebar.number_input("Enter Total Finance Amount", min_value=0.0, step=1.0, help="Budget allocated to Finance")

# Input DataFrame
input_data = pd.DataFrame({
    "Health": [health],
    "Trade": [trade],
    "Finance": [finance]
})

# Load model & scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prediction Button
if st.sidebar.button("🚀 Predict Cluster"):
    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)

    # Cluster Meaning
    choice = {
        0: "⚠️ Not a Priority",
        1: "🌍 Requires Foreign Aid",
        2: "💪 Self-Sufficient (No Aid Required)"
    }

    # Display result
    st.markdown(
        f"""
        <div style="background-color:#F4F6F7; padding:20px; border-radius:10px; text-align:center; box-shadow: 2px 2px 8px #ccc;">
            <h2 style="color:#117A65;">✅ Cluster Prediction</h2>
            <p style="font-size:20px;">The input data belongs to:</p>
            <h3 style="color:#D35400;">{choice.get(prediction[0])}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Optional: visualize input vs cluster centers
    st.subheader("📊 Comparison with Cluster Centers")
    centers = model.cluster_centers_
    features = ["Health", "Trade", "Finance"]

    fig, ax = plt.subplots()
    ax.bar(features, centers[prediction[0]], alpha=0.6, label="Cluster Center")
    ax.bar(features, scaled_data[0], alpha=0.6, label="Your Input")
    ax.set_title("Your Input vs Cluster Center")
    ax.legend()
    st.pyplot(fig)
