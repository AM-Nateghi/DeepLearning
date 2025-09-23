from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import keras
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Load model
# tf.config.set_visible_devices([], "GPU")  # غیرفعال کردن GPU
model = keras.models.load_model("Twittered-Model.keras")

# You may need to load the TextVectorization layer separately if not included in the model
# If included, you can use model.layers[1] or similar to access it
# For this example, we assume it's included in the model


class TextRequest(BaseModel):
    text: str


app = FastAPI()

# (اختیاری) اگر از دامنه دیگر فراخوانی می‌کنی:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# سرو استاتیک
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# Label mapping (adjust if needed)
label_map = {0: "negative", 1: "neutral", 2: "positive"}


@app.post("/predict")
def predict_sentiment(request: TextRequest):
    print(f"Received text: {request.text}")
    # Predict
    preds = model.predict(tf.constant([request.text]))
    label = np.argmax(preds, axis=1)[0]
    result = label_map.get(label, "unknown")
    return {
        "sentiment": result,
        "sentiment_numeric": int(label),
        "probability": round(float(np.array(preds[0]).max()) * 100, 2),
    }
