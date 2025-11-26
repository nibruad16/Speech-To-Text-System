import os
import csv
import json

# === CONFIGURATION ===
# Ensure this matches your path from inspect_data.py
# (Use the path that worked for you in the last step)
DATA_PATH = r"C:\Users\nibru\Documents\Dev\Dev\ML\Speech To Text System\data\cv-corpus-23.0-2025-09-05\am"
TRAIN_TSV = os.path.join(DATA_PATH, "train.tsv")
OUTPUT_FILE = "vocab.json"
    
def build_vocabulary():
    print("📖 Scanning dataset for unique characters...")
    
    if not os.path.exists(TRAIN_TSV):
        print(f"❌ Error: train.tsv not found at {TRAIN_TSV}")
        return

    # Use a set to store unique characters (automatically removes duplicates)
    unique_chars = set()

    try:
        with open(TRAIN_TSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            
            for row in reader:
                sentence = row['sentence']
                # Add every character in the sentence to our set
                for char in sentence:
                    unique_chars.add(char)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Convert to a sorted list so the IDs are always the same
    sorted_chars = sorted(list(unique_chars))
    
    # Create the mapping: { "a": 1, "b": 2, ... }
    # We start at 1 because 0 is usually reserved for "Padding" (empty space)
    vocab = {char: idx + 1 for idx, char in enumerate(sorted_chars)}
    
    # Add special tokens manually
    # <pad> = Padding (to make all sentences same length)
    # <unk> = Unknown character
    vocab["<pad>"] = 0
    vocab["<unk>"] = len(vocab) 

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"✅ Success! Found {len(sorted_chars)} unique characters.")
    print(f"💾 Vocabulary saved to {OUTPUT_FILE}")
    
    # Print a preview
    print("\n--- Vocabulary Preview ---")
    keys = list(vocab.keys())
    print(f"First 10: {keys[:10]}")
    print(f"Last 10:  {keys[-10:]}")

if __name__ == "__main__":
    build_vocabulary()