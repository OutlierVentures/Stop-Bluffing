import csv
import json
import os
import glob
import progressbar
import numpy as np
import sklearn.preprocessing

LANDMARKS_DIR = 'landmarks'
np.random.seed(7)


def load(loso=0):
    """
    TODO: Implement LOSO (Leave one subject out)
    :return:
    :rtype: (x, y) x = input data, y = binary label
    """
    print("Loading data")
    landmark_paths = glob.glob(os.path.join('data', LANDMARKS_DIR, '*.json'))
    landmark_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    nb_samples = len(landmark_paths)
    # 150 frames, 68 landmarks, 3 x, y, z coordinates
    x = np.zeros((nb_samples, 150, 68, 3))
    clip_ids = np.zeros(nb_samples, dtype=np.uint16)
    bar = progressbar.ProgressBar(max_value=nb_samples)
    for i, path in bar(enumerate(landmark_paths)):
        data = json.load(open(path))
        clip_ids[i] = int(os.path.splitext(os.path.basename(path))[0])
        for t, frame in enumerate(data):
            x[i, t, :, :] = np.array(frame[1])

    y = __load_labels__(clip_ids)

    # Normalize input
    x = normalize(x)

    # Unison shuffle
    idx = np.random.permutation(nb_samples)

    return x[idx], y[idx]


def normalize(x):
    nb_samples = x.shape[0]
    x_new = x.reshape((-1, 3))
    norm = sklearn.preprocessing.normalize(x_new)

    return norm.reshape((nb_samples, 150, 68, 3))


def split_data(x, y):
    """
    Split data into training and validation set
    :return:
    """
    training_percent = 0.7

    split_idx = round(training_percent * len(y))

    x_train = x[:split_idx]
    x_val = x[split_idx:]

    y_train = y[:split_idx]
    y_val = y[split_idx:]

    return x_train, y_train, x_val, y_val


def __load_labels__(clip_ids):
    y = np.zeros(len(clip_ids), dtype=np.uint8)
    label_map = {}
    with open('data/bluff_data.csv') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            label_map[int(row['clipId'])] = int(row['isBluffing'])

    for i, clip_id in enumerate(clip_ids):
        y[i] = label_map[clip_id]

    return y
