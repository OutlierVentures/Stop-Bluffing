from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM, Reshape


def mlp():
    model = Sequential()
    model.add(Flatten(input_shape=(150, 68, 3)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    __add_output__(model)

    return model


def cnn_rnn():
    model = Sequential()
    # 150 time steps, 68 * 3 features
    model.add(Reshape((150, 68 * 3), input_shape=(150, 68, 3)))
    model.add(LSTM(64, input_shape=(150, 68 * 3), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, input_shape=(150, 68 * 3), return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(Dense(256, activation='relu'))
    __add_output__(model)

    return model


def lstm():
    model = Sequential()
    model.add(LSTM())
    pass

def __add_output__(model):
    model.add(Dense(1, activation='sigmoid'))