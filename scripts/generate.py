import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, BertModel
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torch

MODEL_PATH = "../models/music_gen_cnn.h5"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

def generate_spectrogram(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = bert_model(**tokens)
    text_embedding = output.last_hidden_state.mean(dim=1).numpy()

    # Generate spectrogram data
    spectrogram_data = model.predict([text_embedding, np.zeros((1, 128, 128, 3))])  # Empty spectrogram
    spectrogram_data = spectrogram_data.reshape((128, 128))

    # Convert to image
    plt.figure(figsize=(2, 2))
    librosa.display.specshow(spectrogram_data, cmap="magma")
    plt.axis("off")
    plt.savefig("generated.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    return "generated.png"

# Convert spectrogram to WAV
def spectrogram_to_wav(spectrogram_path):
    img = Image.open(spectrogram_path).convert("L")
    S_db = np.array(img, dtype=np.float32)
    S = librosa.db_to_power(S_db)
    y = librosa.feature.inverse.mel_to_audio(S)
    librosa.output.write_wav("generated_music.wav", y, sr=22050)

# Run inference
text_prompt = "a soulful calm sound"
generated_spectrogram = generate_spectrogram(text_prompt)
spectrogram_to_wav(generated_spectrogram)

print("Music generated and saved as generated_music.wav!")