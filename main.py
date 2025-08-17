from fastapi import FastAPI
import joblib

app = FastAPI()

pipeline = None
try:
    pipeline = joblib.load("income_pipeline.pkl")
except Exception as e:
    print("⚠️ Warning: Could not load pipeline:", e)

@app.get("/")
def root():
    return {"message": "Income Calculator API is running 🚀"}

@app.post("/predict")
def predict(features: dict):
    if pipeline is None:
        return {"error": "Model not available on server. Please check deployment."}
    try:
        prediction = pipeline.predict([list(features.values())])[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
