import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import json
from utils import process_audio  # We reuse your audio function

class SpeechDataset(Dataset):
    def __init__(self, tsv_file, clips_dir, vocab_file):
        """
        Args:
            tsv_file (str): Path to the train.tsv file
            clips_dir (str): Path to the folder containing mp3s
            vocab_file (str): Path to the vocab.json file
        """
        self.clips_dir = clips_dir
        
        # 1. Load the Data Map (TSV)
        self.data = pd.read_csv(tsv_file, sep='\t')
        
        # 2. Load the Vocabulary
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.char_to_id = json.load(f)

    def __len__(self):
        # Tells PyTorch how many samples we have
        return len(self.data)

    def text_to_int(self, text):
        """
        Converts "ሰላም" -> [55, 102, 33]
        """
        targets = []
        for char in text:
            if char in self.char_to_id:
                targets.append(self.char_to_id[char])
            else:
                # If we find a character not in vocab, use <unk> (Unknown)
                targets.append(self.char_to_id['<unk>'])
        return torch.tensor(targets, dtype=torch.long)

    def __getitem__(self, idx):
        # This function runs every time the model asks for ONE sample
        
        # A. Get file path and text from the row
        row = self.data.iloc[idx]
        audio_filename = row['path']
        text = row['sentence']
        
        audio_path = os.path.join(self.clips_dir, audio_filename)
        
        # B. Process Audio (Spectrogram)
        # Result shape: (Time, 128)
        spectrogram = process_audio(audio_path)
        
        # C. Process Text (Numbers)
        # Result shape: (Length of sentence)
        label = self.text_to_int(text)
        
        # D. Safety Check
        # If audio failed to load (corrupt file), we return None
        if spectrogram is None:
            return None
            
        return spectrogram, label

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test if the Dataset works
    print("🧪 Testing SpeechDataset...")
    
    # paths (Adjust these to match your PC if needed)
    BASE_PATH = r"../data/cv-corpus-23.0-2025-09-05/am"
    TSV = os.path.join(BASE_PATH, "train.tsv")
    CLIPS = os.path.join(BASE_PATH, "clips")
    VOCAB = "vocab.json"
    
    # Initialize
    dataset = SpeechDataset(TSV, CLIPS, VOCAB)
    
    print(f"✅ Dataset Loaded. Total samples: {len(dataset)}")
    
    # Get the first sample
    spec, label = dataset[0]
    print("\n--- Sample 0 ---")
    print(f"📊 Spectrogram Shape: {spec.shape} (Time, Freq)")
    print(f"🔢 Label Tensor: {label}")
    print(f"📝 Raw Numbers: {label.tolist()}")