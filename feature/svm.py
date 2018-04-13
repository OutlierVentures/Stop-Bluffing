"""
SVM Classifier for feature-based model
"""
from sklearn import svm
from sklearn.model_selection import cross_val_score
from feature import load_labelled
from model import evaluate


def train():
    # x, y = loader.load_sum()
    x_train, y_train, x_val, y_val = load_labelled.read_labelled()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    # clf = svm.SVC(class_weight='balanced')
    clf = svm.SVC(class_weight='balanced', kernel='linear')
    clf.fit(x_train, y_train)

    evaluate.evaluate(clf, x_val, y_val)
    # print(clf.score(x_val,y_val))


if __name__ == '__main__':
    train()
