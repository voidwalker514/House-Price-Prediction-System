from pydantic import BaseModel
from typing import List

class HouseFeatures(BaseModel):
    Area_Sqft: float
    Bedrooms: int
    Bathrooms: int
    Location_Score: float
    Age_of_Property: int
    Furnishing_Status: str

class PredictionOutput(BaseModel):
    predicted_price: float
    currency: str = "USD"
