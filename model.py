import tensorflow as tf
import numpy as np
from datasets import load_dataset

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import preprocess_input


dataset = load_dataset("Parveshiiii/AI-vs-Real", verification_mode="no_checks")

base_model = VGG16( # Loading the VGG16 model
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3) # 224x224 is the input size for VGG16, and 3 is for RGB channels
)

base_model.trainable = False # Turning off the training of the base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")
]) # Explains the model 



split_1 = dataset["train"].train_test_split(test_size=0.25, seed=42) # This autmactily from hugginface puts 75% of the data into training and 25% into testing. 
# The seed is used to ensure that the split is reproducible. So "test" didn’t exist before. The split function created that key.

train_data = split_1["train"] # Train data is 75% of the original dataset
temp_data = split_1["test"]  # Test data is 25% of the original dataset, but we will split it further into validation and test sets.

split_2 = temp_data.train_test_split(test_size=0.5, seed=42) # Cut the 25% test data into 50% validation and 50% test data. So now we have 75% training, 12.5% validation, and 12.5% test data.

val_data = split_2["train"] # How well the model is doing on unseen images while training is happening.
test_data = split_2["test"]


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # how model will be trained and show accuracy



def prepare_example(example):
    img = example["image"]              # get the PIL image

    img = img.convert("RGB")            # make sure it has 3 channels
    img = img.resize((224, 224))        

    img = np.array(img, dtype=np.float32)  # pixels -> numbers

    img = preprocess_input(img)         # prepare numbers the way VGG16 expects

    label = example["binary_label"]     # 0 = AI, 1 = real

    return img, label


def make_tf_dataset(hf_data, batch_size=32, shuffle=False):

    def generator():
        for example in hf_data:
            img, label = prepare_example(example)

            # Give TensorFlow this image + label, then pause until it asks for the next one
            yield img, np.float32(label)

    tf_dataset = tf.data.Dataset.from_generator(
        generator,

        output_signature=(
            # 224 high, 224 wide, 3 RGB channels, float numbers
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),

            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(1000)

    tf_dataset = tf_dataset.batch(batch_size)

    return tf_dataset


# Keep the existing model/training logic while fitting within the 4 GB GPU.
train_tf = make_tf_dataset(train_data, batch_size=8, shuffle=True).repeat()
val_tf = make_tf_dataset(val_data, batch_size=8)
test_tf = make_tf_dataset(test_data, batch_size=8)

model.summary()


model.fit(
    train_tf,
    validation_data=val_tf,
    epochs=10,
    steps_per_epoch=len(train_data) // 8
)


model.evaluate(test_tf)
print("accuracy: ", model.evaluate(test_tf)[1])


model.save("Ai_Or_Not.keras")