import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    model_path = 'models/house_price_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Header
st.title("🏠 Real Estate Price Prediction Dashboard")
st.markdown("""
Predict the market value of your property using our advanced Machine Learning model. 
Adjust the features in the sidebar to see real-time price updates.
""")

# Sidebar for inputs
st.sidebar.header("📋 Property Features")

area = st.sidebar.number_input("Total Area (Sqft)", min_value=500, max_value=5000, value=1500, step=50)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 5, 3)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 5, 2)
location_score = st.sidebar.slider("Location Score (1: Rural, 10: Prime)", 1.0, 10.0, 7.5, 0.1)
age = st.sidebar.number_input("Age of Property (Years)", 0, 50, 5)
furnishing = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Market Analysis")
    
    if model:
        # Prepare input data
        input_data = pd.DataFrame({
            'Area_Sqft': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Location_Score': [location_score],
            'Age_of_Property': [age],
            'Furnishing_Status': [furnishing]
        })
        
        # Prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success(f"### Estimated Market Price: ${prediction:,.2f}")
        
        # Comparison with average (mock logic for demo)
        avg_price = 300000
        diff = ((prediction - avg_price) / avg_price) * 100
        st.metric("Price relative to market average", f"${prediction:,.0f}", f"{diff:+.1f}%")
        
        # Feature Impact (Static explanation for student project)
        st.info("💡 **Model Insights:** Area and Location Score have the highest positive impact on the price.")
    else:
        st.error("Model file not found. Please run `python src/train.py` first to train the model.")

with col2:
    st.subheader("🖼️ Visual Insights")
    if os.path.exists('outputs/correlation_heatmap.png'):
        st.image('outputs/correlation_heatmap.png', caption="Feature Correlation")
    if os.path.exists('outputs/actual_vs_predicted.png'):
        st.image('outputs/actual_vs_predicted.png', caption="Model Performance")

# Data Preview
if st.checkbox("Show Sample Dataset"):
    if os.path.exists('data/house_prices.csv'):
        df = pd.read_csv('data/house_prices.csv')
        st.write(df.head(10))
    else:
        st.write("Dataset not found.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Antigravity AI")
