import os
import glob
from tqdm import tqdm
import numpy as np
import librosa

# label mapping (index -> class name)
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
# build inverse mapping
label_names = [None] * (max(label_dict.values()) + 1)
for k, v in label_dict.items():
    label_names[v] = k

def extract_features_for_train(parent_dir, sub_dirs, max_file_per_class=None, file_ext="*.wav"):
    features = []
    labels = []
    for sub_dir in sub_dirs:
        pattern = os.path.join(parent_dir, sub_dir, file_ext)
        files = sorted(glob.glob(pattern))
        if max_file_per_class is not None:
            files = files[:max_file_per_class]
        for fn in tqdm(files, desc=f'Extract {sub_dir}'):
            category = os.path.basename(os.path.dirname(fn))
            if category not in label_dict:
                raise ValueError(f"Unknown category '{category}' from file {fn}")
            y, sr = librosa.load(fn, res_type='kaiser_fast')
            mels = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).T, axis=0)
            features.append(mels)
            labels.append(label_dict[category])
    features = np.array(features, dtype=np.float32)  # (N,128)
    labels = np.array(labels, dtype=np.int64)        # (N,)
    return features, labels

def extract_features_for_test(test_dir, file_ext="*.wav"):
    features = []
    filenames = []
    files = sorted(glob.glob(os.path.join(test_dir, file_ext)))
    for fn in tqdm(files, desc='Extract test'):
        y, sr = librosa.load(fn, res_type='kaiser_fast')
        mels = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).T, axis=0)
        features.append(mels)
        filenames.append(os.path.basename(fn))
    features = np.array(features, dtype=np.float32)
    filenames = np.array(filenames, dtype=object)
    return features, filenames

def generate_train(parent_dir='./train_sample', max_file_per_class=None):
    sub_dirs = ['aloe', 'burger', 'cabbage', 'candied_fruits',
                'carrots', 'chips', 'chocolate', 'drinks', 'fries',
                'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
                'pizza', 'ribs', 'salmon', 'soup', 'wings']
    X, y = extract_features_for_train(parent_dir, sub_dirs, max_file_per_class)
    # 保存为两个文件（推荐）
    np.save('fusai_train_features.npy', X)
    np.save('fusai_train_labels.npy', y)
    # 保存标签名称数组（索引 -> class name）
    np.save('label_names.npy', np.array(label_names, dtype=object))
    print('Saved: fusai_train_features.npy', X.shape)
    print('Saved: fusai_train_labels.npy', y.shape)
    print('Saved: label_names.npy', np.array(label_names).shape)

def generate_test(test_dir='./test_a'):
    X_test, filenames = extract_features_for_test(test_dir)
    np.save('./fusai_test_mfcc_128.npy', X_test)
    np.save('./fusai_test_filenames.npy', filenames)
    print('Saved test features and filenames:', X_test.shape, filenames.shape)

if __name__ == '__main__':
    # 修改路径或 max_file_per_class 根据需要
    generate_train(parent_dir='../train', max_file_per_class=None)
    generate_test(test_dir='../test_a')