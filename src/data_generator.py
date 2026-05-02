import pandas as pd
import numpy as np
import os

def generate_house_data(n_samples=1000):
    np.random.seed(42)
    
    # Features
    area = np.random.normal(1500, 500, n_samples).clip(500, 5000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    bathrooms = bedrooms - np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    bathrooms = np.maximum(bathrooms, 1)
    
    location_score = np.random.uniform(1, 10, n_samples) # 1: Rural, 10: Prime City Center
    age_of_property = np.random.randint(0, 30, n_samples)
    
    furnishing_status = np.random.choice(['Unfurnished', 'Semi-Furnished', 'Fully Furnished'], n_samples)
    furnishing_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully Furnished': 2}
    
    # Target: Price calculation (with some noise)
    # Base price + Area * 100 + Bedrooms * 5000 + Location * 10000 - Age * 2000
    base_price = 50000
    price = (base_price + 
             (area * 150) + 
             (bedrooms * 10000) + 
             (location_score * 20000) - 
             (age_of_property * 1500) + 
             (np.array([furnishing_map[f] for f in furnishing_status]) * 15000) +
             np.random.normal(0, 10000, n_samples))
    
    price = price.clip(50000) # Minimum price 50k
    
    df = pd.DataFrame({
        'Area_Sqft': area.round(0),
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Location_Score': location_score.round(1),
        'Age_of_Property': age_of_property,
        'Furnishing_Status': furnishing_status,
        'Price': price.round(0)
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/house_prices.csv', index=False)
    print(f"Dataset generated with {n_samples} samples and saved to data/house_prices.csv")
    return df

if __name__ == "__main__":
    generate_house_data()
