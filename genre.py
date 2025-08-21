import librosa
import numpy as np

def extract_features(file):
    y, sr = librosa.load(file, duration=30)  # load 30s of audio
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    
    return np.hstack([mfcc, chroma, spec_cent, tempo])

import os
import pandas as pd

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
data = []

for g in genres:
    folder = f"data/genres/{g}"
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(folder, file))
            data.append([features, g])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['features', 'label'])
