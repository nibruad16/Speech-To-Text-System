import librosa
import numpy as np
import torch

def process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, n_fft=2048, hop_length=512)

    mfcc = librosa.power_to_db(mfcc)
    mfcc = mfcc.T

    return torch.tensor(mfcc, dtype=torch.float32)

if __name__ == "__main__":
    print("utils.py is ready.")