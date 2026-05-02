import pandas as pd
import joblib
import os
import time

def run_simulation():
    print("="*50)
    print(" HOUSE PRICE PREDICTION SYSTEM SIMULATION")
    print("="*50)
    
    # 1. Check for model
    if not os.path.exists('models/house_price_model.joblib'):
        print("Training model first...")
        from src.train import train_model
        train_model()
    
    model = joblib.load('models/house_price_model.joblib')
    
    # 2. Simulated user input
    print("\n[STEP 1] Receiving House Details...")
    sample_house = {
        'Area_Sqft': [1800],
        'Bedrooms': [3],
        'Bathrooms': [2],
        'Location_Score': [8.5],
        'Age_of_Property': [5],
        'Furnishing_Status': ['Semi-Furnished']
    }
    input_df = pd.DataFrame(sample_house)
    print(input_df.to_string(index=False))
    
    # 3. Model Inference
    print("\n[STEP 2] Running ML Inference Pipeline...")
    time.sleep(1)
    prediction = model.predict(input_df)[0]
    
    # 4. Output
    print("\n" + "!"*50)
    print(f"PREDICTED MARKET PRICE: ${prediction:,.2f}")
    print("!"*50)
    
    print("\n[STEP 3] Model Insights:")
    print("- Higher Area significantly increases price (+ $150/sqft)")
    print("- Prime location score (8.5/10) adds premium value")
    print("- Low property age (5 yrs) maintains high valuation")
    
    print("\nCheck 'outputs/' folder for visualization graphs!")

if __name__ == "__main__":
    run_simulation()
