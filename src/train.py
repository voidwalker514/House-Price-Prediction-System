import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_model():
    # 1. Load Data
    df = pd.read_csv('data/house_prices.csv')
    print("Data loaded successfully.")
    
    # 2. EDA & Visualization
    os.makedirs('outputs', exist_ok=True)
    
    # Correlation Heatmap (only for numeric columns)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    
    # Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, color='blue')
    plt.title('Distribution of House Prices')
    plt.savefig('outputs/price_distribution.png')
    plt.close()

    # 3. Feature Engineering & Preprocessing
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    categorical_features = ['Furnishing_Status']
    numeric_features = ['Area_Sqft', 'Bedrooms', 'Bathrooms', 'Location_Score', 'Age_of_Property']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Model Selection & Training
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"{name} Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_model_name = name

    # 6. Save the Best Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/house_price_model.joblib')
    print(f"Best model ({best_model_name}) saved to models/house_price_model.joblib")
    
    # 7. Final Visualization: Actual vs Predicted
    y_pred_final = best_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_final, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted Price ({best_model_name})')
    plt.savefig('outputs/actual_vs_predicted.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    train_model()
