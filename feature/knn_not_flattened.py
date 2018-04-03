from feature import loader
from sklearn import neighbors
from model import evaluate

import numpy as np
import csv

DATA_PATH = 'data/labelled_au.csv'

def load():
    nb_lines = len(open(DATA_PATH).readlines()) - 1  # Subtract 1 for header
    nb_samples = nb_lines
    x = np.zeros((nb_samples, 35))
    y = np.zeros(nb_samples)

    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            y[i] = row['isBluffing']
            x[i] = list(row.values())[0:35]

    return x, y

if __name__ == "__main__":
    x, y = load()
    x_train, y_train, x_val, y_val = loader.split_data(x, y)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    knn = neighbors.KNeighborsClassifier()
    knn.fit(x_train, y_train)

    evaluate.evaluate(knn, x_val, y_val)
