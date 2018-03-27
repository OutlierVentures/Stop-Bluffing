import argparse
from keras import losses
from keras import callbacks

from model import architecture, loader, evaluate


def get_model(name):
    if name == 'mlp':
        return architecture.mlp()
    if name == 'cnn_rnn':
        return architecture.cnn_rnn()
    else:
        raise ValueError('Model {} is not defined'.format(name))


def train(args):
    x, y = loader.load()
    x_train, y_train, x_val, y_val = loader.split_data(x, y)

    model = get_model(args.model)

    print(model.summary())

    model.compile(optimizer='adam',
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=5)

    model.fit(x_train, y_train,
              epochs=50,
              batch_size=16,
              validation_data=(x_val, y_val),
              callbacks=[early_stopping])

    # Print evaluation matrix
    evaluate.evaluate(model, x_val, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    train(args)
