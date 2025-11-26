import os
import csv
import librosa

data_path = r"C:\Users\nibru\Documents\Dev\Dev\ML\Speech To Text System\data\cv-corpus-23.0-2025-09-05\am"
clips_path = os.path.join(data_path, "clips")
train_path = os.path.join(data_path, "train.tsv")


def inspect_data():

    if not os.path.exists(clips_path):
        print(f"Clips path does not exist: {clips_path}")
        return
    if not os.path.exists(train_path):
        print(f"Train path does not exist: {train_path}")
        return
    
    print("Data paths verified.")


try:
    with open(train_path, "r", newline="", encoding="utf-8") as tsvfile:

        reader = csv.DictReader(tsvfile, delimiter = "\t")

        count = 0
        for row in reader:
            if count  >= 3:
                break
                
            filename = row.get("path")
            text = row.get("sentence")

            audio_file_path = os.path.join(clips_path, filename)

            try:
                if os.path.exists(audio_file_path):
                    y, sr = librosa.load(audio_file_path, sr= 16000 , duration=1.0)

                    print(f"  Audio verified! Shape: {y.shape}")

                else:

                    print(f" File missing: {audio_file_path}")
            except Exception as e:

                print(f" Error loading audio file {audio_file_path}: {e}")

            count += 1
except Exception as e:
    print(f" Error reading train file {train_path}: {e}")


if __name__ == "__main__":
    inspect_data()











