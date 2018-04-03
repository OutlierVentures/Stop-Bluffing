from time import time

import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Permute, Reshape
from keras import backend as K

from random import randint
import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
import itertools

FRAMES_PER_CLIP = 150

def read_labelled(au_type="c", filename="labelled.csv"):
	labelled = pandas.read_csv(filename, sep='\s*,\s*', engine='python')

	player_out = randint(0,5)
	loso = labelled[labelled.playerId != player_out]
	loso = loso.reset_index(drop=True)

	au_samples = []
	isBluffing_samples = []
	isBluffing_df = loso['isBluffing']

	au_feats = loso.filter(regex = '(confidence)|AU.*_'+au_type, axis=1)

	num_frames = au_feats.shape[0]
	num_samples = int(num_frames / FRAMES_PER_CLIP)

	for sample_idx in range(num_samples):
		start_frame = sample_idx * 150
		end_frame = start_frame + 149
		sample_feat = au_feats.loc[start_frame:end_frame, :].as_matrix()
		sample_isBluffing = isBluffing_df.loc[start_frame]
		au_samples.append(sample_feat)
		isBluffing_samples.append(sample_isBluffing)
	return au_samples, isBluffing_samples

def split_dataset(x_dataset, y_dataset, ratio):
	split_idx = int(ratio * len(x_dataset))
	x_split0 = np.array(x_dataset[:split_idx])
	y_split0 = np.array(y_dataset[:split_idx])
	x_split1 = np.array(x_dataset[split_idx:])
	y_split1 = np.array(y_dataset[split_idx:])
	return x_split0, y_split0, x_split1, y_split1

def cnn_rnn(input_shape):
    # nb_frames, nb_landmarks, _ = input_shape
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

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')



if __name__ == "__main__":
	x_dataset, y_dataset = read_labelled()
	# y_dataset = keras.utils.to_categorical(y_dataset)
	training_ratio = 0.8
	validation_ratio = 0.1
	test_ratio = 0.1
	x_train, y_train, x_test, y_test = split_dataset(x_dataset, y_dataset, 0.8)
	x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, 0.9)
	print("training samples")
	print(len(x_train))
	print("validation samples")
	print(len(x_val))
	print("test samples")
	print(len(x_test))

	frames, features = x_train[0].shape
	print(x_train.shape[1:])
	model = Sequential()

	# model.add(Flatten(input_shape=x_train.shape[1:]))
	# model.add(Dense(units=1024, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(units=1024, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(units=1024, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(1, activation='sigmoid'))

	# model = cnn_rnn(x_train.shape[1:])
	model.add(LSTM(1024, return_sequences=True, input_shape=x_train.shape[1:]))
	model.add(LSTM(1024, return_sequences=False))
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1,activation='sigmoid'))

	epochs = 5
	batch_size = 32

	model.compile(optimizer='adam',
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])
	model.summary()

	training_start = time()
	print("starting training")

	H = model.fit(x_train, y_train,
				  batch_size=batch_size,
				  epochs=epochs,
				  verbose=1,
				  shuffle=True,
				  validation_data=(x_val, y_val))

	training_time = time() - training_start
	print("training finished")
	model.summary()

	inference_start = time()
	loss, accuracy = model.evaluate(x=x_test, y=y_test)
	print(loss, accuracy)
	from sklearn.metrics import confusion_matrix
	y_pred = model.predict_classes(x=x_test)
	con_mat = confusion_matrix(y_test,y_pred)

	class_names = ['no bluffing','bluffing']

	# plt.figure()
	# plot_confusion_matrix(con_mat, classes=class_names,
	#                       title='Confusion matrix, without normalization')

	# plt.figure()
	# plot_confusion_matrix(con_mat, classes=class_names, normalize=True,
	#                       title='Normalized confusion matrix')

	# plt.show()
	# inference_time = time() - inference_start
