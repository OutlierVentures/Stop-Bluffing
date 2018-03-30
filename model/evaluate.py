import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test).flatten()
    print(y_pred)
    y_pred = np.round(y_pred).astype(np.uint8)

    cnf_matrix = confusion_matrix(y_test, y_pred)

    f, (ax1, ax2) = plt.subplots(1, 2)
    plot_confusion_matrix(cnf_matrix,
                          classes=['Not bluffing', 'Bluffing'],
                          axes=ax1,
                          title='Confusion matrix, without normalization')

    plot_confusion_matrix(cnf_matrix,
                          classes=['Not bluffing', 'Bluffing'],
                          axes=ax2,
                          normalize=True,
                          title='Confusion matrix, normalized')

    plt.show()


def plot_confusion_matrix(cm, classes, axes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.sca(axes)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')