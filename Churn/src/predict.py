import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

RUN_ID = "36f80842a35d4c3386335c34d4c17942" 
LOGGED_MODEL = f"runs:/{RUN_ID}/model"

app = FastAPI(title="Churn Predictor API")

print(f"Loading model from {LOGGED_MODEL}...")
try:
    model = mlflow.pyfunc.load_model(LOGGED_MODEL)
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. Did you set the RUN_ID? Details: {e}")
    model = None

class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_df = pd.DataFrame([data.dict()])
    
    try:
        prediction = model.predict(input_df)
        # prediction is an array like [0] or [1]
        result = int(prediction[0])
        return {
            "churn_prediction": result, 
            "message": "Customer likely to churn" if result == 1 else "Customer safe"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
def root():
    return {"status": "Model is live"}