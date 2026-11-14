import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


dataset = load_dataset("Parveshiiii/AI-vs-Real", verification_mode="no_checks")

# Load the pretrained VGG16 feature extractor.
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

# Add the binary classification head.
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid"),
])


split_1 = dataset["train"].train_test_split(test_size=0.25, seed=42)
train_data = split_1["train"]
temp_data = split_1["test"]

split_2 = temp_data.train_test_split(test_size=0.5, seed=42)
val_data = split_2["train"]
test_data = split_2["test"]


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


def prepare_example(example):
    img = example["image"].convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    label = example["binary_label"]
    return img, label


def make_tf_dataset(hf_data, batch_size=32, shuffle=False):
    def generator():
        for example in hf_data:
            img, label = prepare_example(example)
            yield img, np.float32(label)

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(1000)

    return tf_dataset.batch(batch_size)


train_tf = make_tf_dataset(train_data, batch_size=8, shuffle=True).repeat()
val_tf = make_tf_dataset(val_data, batch_size=8)
test_tf = make_tf_dataset(test_data, batch_size=8)

model.summary()

model.fit(
    train_tf,
    validation_data=val_tf,
    epochs=10,
    steps_per_epoch=len(train_data) // 8,
)

print("accuracy: ", model.evaluate(test_tf)[1])
model.save("model/Ai_Or_Not.keras")
