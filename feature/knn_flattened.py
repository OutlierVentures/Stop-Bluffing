from feature import loader, load_labelled
from sklearn import neighbors
from model import evaluate

if __name__ == "__main__":
    # x, y = loader.load_sum()
    # x_train, y_train, x_val, y_val = loader.split_data(x, y)

    x_train, y_train, x_val, y_val = load_labelled.read_labelled()

    print(x_train.shape)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    knn = neighbors.KNeighborsClassifier(weights='distance')
    knn.fit(x_train, y_train)

    evaluate.evaluate(knn, x_val, y_val)
