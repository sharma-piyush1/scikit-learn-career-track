from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the saved pipeline (preprocessor + model)
model = joblib.load("titanic_survival_model.pkl")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts passenger survival using a trained RandomForest + preprocessing pipeline",
    version="1.0"
)

# Input schema for prediction requests
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert input to DataFrame
    df = pd.DataFrame([passenger.dict()])
    
    # Predict using loaded pipeline
    pred = model.predict(df)[0]
    
    return {
        "survived": bool(pred),
        "prediction_raw": int(pred)
    }


