"""
SVM Classifier
"""
import argparse
from sklearn import svm

from model import loader, evaluate, preprocessing


def train(use_fisher):
    x, y = loader.load()

    if use_fisher:
        x = preprocessing.to_fisher(x)

    x_train, y_train, x_val, y_val = loader.split_data(x, y)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    clf = svm.SVC(class_weight='balanced')
    clf.fit(x_train, y_train)

    evaluate.evaluate(clf, x_val, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-fisher', help='Convert input to Fisher encoding', action='store_true')
    args = parser.parse_args()
    train(args.use_fisher)
