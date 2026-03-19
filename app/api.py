from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ✅ CORS (restrict in production later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model + columns
model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")

# ✅ Define request schema (VERY IMPORTANT 🔥)
class HouseInput(BaseModel):
    space: float
    rooms: int
    floors: int
    location: str


@app.get("/")
def home():
    return {"message": "House Price Prediction API Running 🚀"}


@app.post("/predict")
def predict(data: HouseInput):
    try:
        # Convert input to dict → DataFrame
        df = pd.DataFrame([data.dict()])

        # One-hot encoding
        df = pd.get_dummies(df)

        # Match training columns
        df = df.reindex(columns=columns, fill_value=0)

        # Prediction
        prediction = model.predict(df)

        return {
            "predicted_price": float(prediction[0]),
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }