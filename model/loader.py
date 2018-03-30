import csv
import json
import os
import math
import glob
import progressbar
import numpy as np
import sklearn.preprocessing

LANDMARKS_DIR = 'landmarks'
np.random.seed(7)


def load(shuffle=True, loso=0):
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
    x = np.zeros((nb_samples, 150, 51, 3))
    clip_ids = np.zeros(nb_samples, dtype=np.uint16)
    bar = progressbar.ProgressBar(max_value=nb_samples)
    for i, path in bar(enumerate(landmark_paths)):
        data = json.load(open(path))
        if len(data) == 0:
            raise ValueError
        clip_ids[i] = int(os.path.splitext(os.path.basename(path))[0])
        for t, frame in enumerate(data):
            # Discard lower face landmarks to reduce dimensionality by skipping idx 0 to 16
            x[i, t, :, :] = np.array(frame[1])[17:, :]

    y = __load_labels__(clip_ids)

    # Rescale input
    x = rescale(x)

    # Unison shuffle
    if shuffle:
        idx = np.random.permutation(nb_samples)
        return x[idx], y[idx]

    return x, y


def rescale(x):
    """
    Rescales input to be in range 0 to 1 to avoid over/underflow
    :param x:
    :return: Rescaled x
    """
    nb_samples, t, nb_landmarks, _ = x.shape
    x_new = x.reshape((-1, 3))
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_new)
    x_new = scaler.transform(x_new)

    return x_new.reshape((nb_samples, t, nb_landmarks, 3))


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


def compact_frames(x, window_size=5, step_size=4):
    """
    Compact time frames by applying a moving average filter
    :param x: The input vector of shape (nb_samples, t, n, d)
    :param window_size: Size of the window to average
    :param step_size: Sample every nth element from moving average
    :return: The compacted input vector, with averaged input frames
    """
    nb_samples, t, n, d = x.shape

    # Flatten x,y,z
    x_flat = x.reshape((nb_samples, t, n * d))
    compact_t = math.ceil((t - window_size + 1) / step_size)
    x_compact = np.zeros((nb_samples, compact_t, n, d))

    # Apply moving average window
    conv_filter = np.ones((window_size,)) / window_size
    for i in range(nb_samples):
        running_avg = np.apply_along_axis(
            lambda row: np.convolve(row, conv_filter, mode='valid'), axis=0, arr=x_flat[i, :, :])
        x_compact[i, :, :, :] = running_avg[0::step_size, :].reshape((compact_t, n, d))

    return x_compact


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
