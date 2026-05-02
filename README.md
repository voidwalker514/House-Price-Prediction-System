# House Price Prediction using Regression Models 🏠💰

![Banner](https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80)

## 📌 Project Overview
This project is an end-to-end Machine Learning solution designed to predict house sale prices based on various property features such as area, location, age, and furnishing status. It includes a complete ML pipeline, a FastAPI backend, and a modern dashboard.

## 🚀 Key Features
- **Synthetic Data Generation**: Realistic housing dataset creation for experimentation.
- **EDA & Insights**: Comprehensive data analysis with correlation heatmaps and distribution plots.
- **Model Comparison**: Evaluation of Linear Regression and Random Forest models.
- **Production-Ready API**: FastAPI backend for real-time price inference.
- **Interactive Dashboard**: Next.js mini-dashboard for user interaction (Visual Simulation).

## 🛠️ Tech Stack
- **Languages**: Python, JavaScript
- **ML Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Next.js, Tailwind CSS (optional), Vanilla CSS
- **Deployment**: Joblib for model serialization

## 📊 Dataset Features
| Feature | Description |
| :--- | :--- |
| **Area_Sqft** | Total living area in square feet |
| **Bedrooms** | Number of bedrooms |
| **Bathrooms** | Number of bathrooms |
| **Location_Score** | Quality of location (1-10) |
| **Age_of_Property** | Age of the house in years |
| **Furnishing_Status**| Unfurnished, Semi-Furnished, Fully Furnished |

## 📈 Model Performance
| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | ~7,400 | ~9,200 | **0.989** |
| **Random Forest** | ~12,800 | ~16,500| 0.966 |

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Set up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
```bash
# Generate data and train model
python src/data_generator.py
python src/train.py

# Start the API
python src/api.py

# Start the Streamlit Dashboard
streamlit run app.py
```

## 🖼️ Screenshots
*(Include your saved charts from `outputs/` folder here)*
- Correlation Heatmap
- Price Distribution
- Actual vs Predicted Plot

## 🎓 Learning Outcomes
- Implementing regression models from scratch.
- Handling categorical data using One-Hot Encoding.
- Building and serving a Machine Learning API.
- Creating professional documentation for GitHub.

---
**Developed by Ishwari Belhekar**  
*Looking for Data Science / ML Internships!*
