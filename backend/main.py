from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import re
import os

app = FastAPI()

# Загружаем модель и векторизатор
model = joblib.load("models/DecisionTreeClassifier_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

class TextIn(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Backend is running"}

@app.post("/predict")
def predict(data: TextIn):
    text = data.text
    text = re.sub(r'[^\w\s]', '', text.lower())
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return {"prediction": int(prediction[0])}

# Главное: указываем PORT для Railway
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
