from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM, Reshape, Conv2D, \
    BatchNormalization, Permute


def mlp(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    __add_output__(model)

    return model


def mlp_fisher(input_shape):
    """
    MLP variant which takes Fisher Vector as input
    :param input_shape:
    :return:
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    __add_output__(model)

    return model

def cnn_rnn(input_shape):
    nb_frames, nb_landmarks, _ = input_shape
    model = Sequential()
    # model.add(Reshape((nb_frames, nb_landmarks * 3), input_shape=input_shape))

    # TODO: Use Conv2D with landmark_idx as channel
    model.add(Permute((1, 3, 2), input_shape=input_shape))
    # Kernel_size 5 frames, along single landmark, with 3 corresponding to x,y z<
    model.add(Conv2D(filters=8, kernel_size=(5, 3),
                     strides=(2, 1),
                     padding='same',
                     activation='relu',
                     data_format='channels_last'))

    _, a, b, c = model.layers[-1].output_shape
    model.add(Reshape((a, -1)))  # Flatten
    model.add(LSTM(1024, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(1024))
    model.add(BatchNormalization())
    __add_output__(model)

    return model


def __add_output__(model):
    model.add(Dense(1, activation='sigmoid'))
