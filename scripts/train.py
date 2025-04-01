import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import BertTokenizer, BertModel
from PIL import Image
import torch
import os

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
METADATA_FILE = "../metadata.csv"
SPECTROGRAM_PATH = "../spectrograms/"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Load dataset
def load_data():
    df = pd.read_csv(METADATA_FILE)
    images, text_embeddings = [], []

    for _, row in df.iterrows():
        # Load spectrogram
        img_path = os.path.join(SPECTROGRAM_PATH, row["filename"].replace(".wav", ".png"))
        img = Image.open(img_path).resize(IMG_SIZE)
        images.append(np.array(img) / 255.0)

        # Convert text to embedding
        text = row["text_description"]
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = bert_model(**tokens)
        text_embeddings.append(output.last_hidden_state.mean(dim=1).numpy())

    return np.array(images), np.array(text_embeddings)

# Load data
X_images, X_text = load_data()

# Build model
image_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image_input")
x = layers.Conv2D(32, (3, 3), activation="relu")(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)

text_input = layers.Input(shape=(768,), name="text_input")
combined = layers.concatenate([text_input, x])
x = layers.Dense(256, activation="relu")(combined)
x = layers.Dense(128, activation="relu")(x)
output = layers.Dense(16384, activation="tanh")(x)  # Generates spectrogram features

model = models.Model(inputs=[text_input, image_input], outputs=output)
model.compile(optimizer="adam", loss="mse")
model.summary()

# Train model
model.fit([X_text, X_images], X_images.reshape(X_images.shape[0], -1), epochs=EPOCHS, batch_size=BATCH_SIZE)
model.save("../models/music_gen_cnn.h5")

print("Training complete! Model saved.")