"""
Load featured-based data
"""
import csv

import numpy as np

DATA_PATH = 'data/labelled_au.csv'
FRAMES_PER_CLIP = 150

np.random.seed(7)


def load_sum():
    """
    Sum binary features across time axis
    :return:
    """
    x, y = load()

    binary_x = x[:, :, 17:35]

    sum_x = np.sum(binary_x, axis=1)

    # Normalise by number of frames
    sum_x /= FRAMES_PER_CLIP

    return sum_x, y


def load(shuffle=True):
    nb_lines = len(open(DATA_PATH).readlines()) - 1  # Subtract 1 for header
    nb_samples = nb_lines // FRAMES_PER_CLIP
    x = np.zeros((nb_samples, 150, 35))
    y = np.zeros(nb_samples)
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            sample_idx = i // FRAMES_PER_CLIP
            frame_idx = i % FRAMES_PER_CLIP

            if frame_idx == 0:
                y[sample_idx] = row['isBluffing']

            x[sample_idx, frame_idx] = list(row.values())[0:35]

    # Unison shuffle
    if shuffle:
        print("Shuffling data")
        idx = np.random.permutation(nb_samples)
        return x[idx], y[idx]

    return x, y


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
