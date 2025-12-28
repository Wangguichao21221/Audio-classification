import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from  tqdm import tqdm
def pre_process_wav(input_dir, output_dir, duration=4, sr=44100):
    os.makedirs(output_dir, exist_ok=False)
    target_sample = int(duration * sr)

    for filename in tqdm(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        sound, sample_rate = librosa.load(filepath, sr=None)

        # Ensure the right sample rate
        assert sample_rate == sr, f"Unexpected sample rate in {filename}"

        # Transform samples to the same length
        if len(sound) > target_sample:
            sound = sound[:target_sample]
        else:
            sound = np.pad(sound, (0, target_sample - len(sound)), mode="constant")

        mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40, hop_length=512).T

        if mfcc.shape[0] > 344:
            mfcc = mfcc[:344]
        else:
            mfcc = np.pad(mfcc, ((0, 344 - mfcc.shape[0]), (0, 0)), mode="constant")

        npy_name = os.path.splitext(filename)[0] + '.npy'
        np.save(os.path.join(output_dir, npy_name), mfcc)

def pre_process_wav_flat(input_dir, output_dir, duration=4, sr=44100):
    os.makedirs(output_dir, exist_ok=False)
    target_sample = int(duration * sr)

    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        for filename in tqdm(os.listdir(class_path)):
            if not filename.lower().endswith('.wav'):
                continue

            filepath = os.path.join(class_path, filename)
            sound, sample_rate = librosa.load(filepath, sr=None)
            assert sample_rate == sr, f"Unexpected sample rate in {filepath}"

            if len(sound) > target_sample:
                sound = sound[:target_sample]
            else:
                sound = np.pad(sound, (0, target_sample - len(sound)), mode="constant")

            mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40, hop_length=512).T

            if mfcc.shape[0] > 344:
                mfcc = mfcc[:344]
            else:
                mfcc = np.pad(mfcc, ((0, 344 - mfcc.shape[0]), (0, 0)), mode="constant")

            npy_name = f"{class_folder}__{os.path.splitext(filename)[0]}.npy"
            np.save(os.path.join(output_dir, npy_name), mfcc)


class MFCCDataset(Dataset):
    def __init__(self, data_dir, is_test=False):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.is_test = is_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        data = np.load(os.path.join(self.data_dir, file_name))
        data = torch.tensor(data, dtype=torch.float32)

        if self.is_test:
            # In test datasets, there's no class name to extract
            return data, file_name
        else:
            # Extract class name from the filename for training/validation datasets
            if "__" in file_name:  # Ensure the file name has the correct format
                class_name, _ = file_name.split("__", 1)
            else:
                raise ValueError(f"Unexpected file name format: {file_name}")
            return data, class_name
if __name__ == "__main__":
    out_train_dir = './train_npy'
    out_test_dir = './test_npy'
    pre_process_wav_flat('../train',out_train_dir)
    pre_process_wav('../test_a',out_test_dir)