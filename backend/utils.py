import librosa
import numpy as np
import torch
import os

def audio_processing(audio_path):

    try:
        y,sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, n_fft=2048, hop_length=512)

    mfcc = librosa.power_to_db(mfcc)

    mfcc = mfcc.T

    return torch.tensor(mfcc, dtype=torch.float32)

if __name__ == "__main__":
   
    print("Testing process_audio function...")
    

    import soundfile as sf
    fake_audio = np.random.uniform(-1, 1, 16000) # 1 second of noise
    sf.write('test_noise.wav', fake_audio, 16000)
    
    result =audio_processing('test_noise.wav')
    
    if result is not None:
        print(f"✅ Success! Input audio: 1 second")
        print(f"📊 Output Matrix Shape: {result.shape}")
        print("   Expected: (Time Steps, 128 Features)")