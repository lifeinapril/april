import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

DATASET_PATH = "../dataset/"
SPECTROGRAM_PATH = "../spectrograms/"
METADATA_FILE = "../metadata.csv"

os.makedirs(SPECTROGRAM_PATH, exist_ok=True)

# Load metadata
df = pd.read_csv(METADATA_FILE)

def wav_to_spectrogram(file_name):
    file_path = os.path.join(DATASET_PATH, file_name)
    save_path = os.path.join(SPECTROGRAM_PATH, file_name.replace(".wav", ".png"))
    
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(2, 2))
    librosa.display.specshow(S_db, sr=sr, cmap="magma")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# Process all WAV files
for _, row in df.iterrows():
    wav_to_spectrogram(row["filename"])

print("Spectrograms generated successfully!")