import librosa
import numpy as np
import os
import csv

# Define path for WAV files
wav_path = "data/WAV_files/"
output_csv = 'results/librosa_features.csv'

# Initialize list of features to extract with Librosa
features_list = []

# Dictionary to map substrings to labels
label_map = {
    'guitar': 'guitar',
    'gtr': 'guitar',
    'piano': 'piano',
    'pno': 'piano',
    'violin': 'violin',
    'vln': 'violin',
    'cello': 'cello',
    'harpsi': 'harpsichord',
    'gongs': 'gongs',
    'bass': 'bass',
    'marimba': 'marimba',
    'oboe': 'oboe',
    'shakuhachi': 'shakuhachi',
    'sitar': 'sitar',
    'flute': 'flute',
    'sax': 'saxophone',
    'trumpet': 'trumpet',
    'viola': 'viola'
}

# Loop through each file in wav_path
for filename in os.listdir(wav_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(wav_path, filename)
        y, sr = librosa.load(file_path)
        
        # Extract Librosa features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        # Average the features across time to get a single vector
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        
        # Put features into an array
        features = np.concatenate((mfcc_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean))
        
        # Extract the label from the file name using the dictionary
        lower_file_name = filename.lower()
        label = None
        for keyword, mapped_label in label_map.items():
            if keyword in lower_file_name:
                label = mapped_label
                break

        # Append the features and label to the list if a label was found
        if label:
            features_list.append(np.append(features, label))


# Write the features to a CSV file without headers
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(features_list)