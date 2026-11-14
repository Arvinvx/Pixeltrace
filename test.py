import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

model = tf.keras.models.load_model("Ai_Or_Not.keras")

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

run_model("test.png")  # pure ai
run_model("test1.png") # ai
run_model("test2.jpg") # real
run_model("test3.jpg") # real
run_model("test4.png") # ai real
run_model("test5.png") # pure ai