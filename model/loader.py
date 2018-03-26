import csv
import json
import os
import glob
import progressbar
import numpy as np

LANDMARKS_DIR = 'landmarks'


def load():
    """

    :return:
    :rtype: (x, y) x = input data, y = binary label
    """
    landmark_paths = glob.glob(os.path.join('data', LANDMARKS_DIR, '*.json'))
    landmark_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    nb_samples = len(landmark_paths)
    # 150 frames, 68 landmarks, 3 x, y, z coordinates
    x = np.zeros((nb_samples, 150, 68, 3))
    clip_ids = np.zeros(nb_samples, dtype=np.uint16)
    bar = progressbar.ProgressBar()
    for i, path in bar(enumerate(landmark_paths)):
        data = json.load(open(path))
        clip_ids[i] = int(os.path.splitext(os.path.basename(path))[0])
        for t, frame in enumerate(data):
            x[i, t, :, :] = np.array(frame[1])

    y = __load_labels__(clip_ids)

    return x, y


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
