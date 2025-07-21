from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import os

app = FastAPI()

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пути к моделям
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models", "DecisionTreeClassifier_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

# Очистка текста
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()

# Pydantic модель
class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Backend is running"}

#@app.get("/", include_in_schema=False)
#def root():
#    return {"message": "✅ API is running"}

@app.post("/predict")
def predict(input: TextInput):
    clean = clean_text(input.text)
    vect = vectorizer.transform([clean])
    prediction = model.predict(vect)
    return {"predicted_class": int(prediction[0])}

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
