from sklearn import svm

from model import loader, evaluate


def train():
    x, y = loader.load()
    x_train, y_train, x_val, y_val = loader.split_data(x, y)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    evaluate.evaluate(clf, x_val, y_val)


if __name__ == '__main__':
    train()
