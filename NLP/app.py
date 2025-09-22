from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Load model
tf.config.set_visible_devices([], "GPU")  # غیرفعال کردن GPU
model = tf.keras.models.load_model("Twittered-Model.keras")

# You may need to load the TextVectorization layer separately if not included in the model
# If included, you can use model.layers[1] or similar to access it
# For this example, we assume it's included in the model


class TextRequest(BaseModel):
    text: str


app = FastAPI()

# Label mapping (adjust if needed)
label_map = {0: "negative", 1: "neutral", 2: "positive"}


@app.post("/predict")
def predict_sentiment(request: TextRequest):
    # Prepare input as batch
    input_text = np.array([request.text])
    # Predict
    preds = model.predict(input_text)
    label = np.argclmax(preds, axis=1)[0]
    result = label_map.get(label, "unknown")
    return {"sentiment": result}
