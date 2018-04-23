"""
Train model which uses facial geometry as input

```
python train.py [model]
```
"""
import argparse
from collections import Counter
from keras import losses
from keras import callbacks
from model import architecture, loader, evaluate, preprocessing


def get_model(name, input_shape):
    """"
    Retrieve model architecture depending on cmd args
    """
    if name == 'mlp':
        return architecture.mlp(input_shape)
    if name == 'mlp_fv':
        return architecture.mlp_fisher(input_shape)
    if name == 'cnn_rnn':
        return architecture.cnn_rnn(input_shape)
    raise ValueError('Model {} is not defined'.format(name))


def get_class_weights(y):
    """
    Determine class weights based on frequency distribution of labels

    :param y:
    :return:
    """
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority/count) for cls, count in counter.items()}


def train(args):
    x, y = loader.load()

    if args.model == 'mlp_fv':
        x = preprocessing.to_fisher(x)
        nb_samples, nb_landmarks, l = x.shape
        input_shape = (nb_landmarks, l)
    else:
        nb_samples, nb_frames, nb_landmarks, _ = x.shape
        input_shape = (nb_frames, nb_landmarks, 3)
    # x = loader.compact_frames(x, window_size=5, step_size=2)
    x_train, y_train, x_val, y_val = loader.split_data(x, y)

    model = get_model(args.model, input_shape=input_shape)

    print("Input shape: {}".format(x.shape))
    print(model.summary())

    model.compile(optimizer='adam',
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    checkpointer = callbacks.ModelCheckpoint(filepath="data/weights.hdf5", verbose=1, save_best_only=True)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2)

    class_weights = get_class_weights(y_train)
    model.fit(x_train, y_train,
              epochs=50,
              batch_size=8,
              validation_data=(x_val, y_val),
              callbacks=[
                  checkpointer,
                  early_stopping,
              ],
              class_weight=class_weights,
              )

    # Load best model
    model.load_weights('data/weights.hdf5')

    # Print evaluation matrix
    train_score = model.evaluate(x_train, y_train)
    val_score = model.evaluate(x_val, y_val)

    print(model.metrics_names, train_score, val_score)

    evaluate.evaluate(model, x_train, y_train)
    evaluate.evaluate(model, x_val, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    train(args)
