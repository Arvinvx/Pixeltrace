import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


model = tf.keras.models.load_model("model/Ai_Or_Not.keras")


def run_model(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    print("raw prediction:", prediction)

    if prediction >= 0.5:
        print("REAL")
    else:
        print("AI")


run_model("samples/ai_01.png")
run_model("samples/ai_02.png")
run_model("samples/real_01.jpg")
run_model("samples/real_02.jpg")
run_model("samples/uncertain_01.png")
run_model("samples/ai_03.png")
