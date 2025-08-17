from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Income Classifier API")

# Allow all origins for CORS (React frontend will need this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained pipeline
pipeline = joblib.load("income_pipeline.pkl")

# Input data schema (match the dataset columns exactly)
class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = pipeline.predict(df)[0]
    label = ">50K" if int(pred) == 1 else "<=50K"
    proba = None
    try:
        proba = float(pipeline.predict_proba(df)[0][1])
    except Exception:
        pass
    return {"prediction": label, "prob_gt_50k": proba}
