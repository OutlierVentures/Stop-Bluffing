from time import time

import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Permute, Reshape
from keras import backend as K
from collections import Counter

import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
import itertools

FRAMES_PER_CLIP = 150

def read_labelled(au_type="r", filename="labelled.csv"):
	labelled = pandas.read_csv(filename, sep='\s*,\s*', engine='python')
	au_samples = []
	isBluffing_samples = []
	isBluffing_df = labelled['isBluffing']
	au_feats = labelled.filter(regex = '(confidence)|AU.*_'+au_type, axis=1)
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

def get_class_weights(y):
    """
    Determine class weights based on frequency distribution of labels
    :param y:
    :return:
    """
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority/count) for cls, count in counter.items()}

if __name__ == "__main__":
	x_dataset, y_dataset = read_labelled()
	x_train, y_train, x_val, y_val = split_dataset(x_dataset, y_dataset, 0.7)
	print("training samples")
	print(len(x_train))
	print("validation samples")
	print(len(x_val))

	frames, features = x_train[0].shape
	print(x_train.shape[1:])

	# # Keras
	# model = Sequential()
	# # model.add(Flatten(input_shape=x_train.shape[1:]))
	# # model.add(Dense(units=1024, activation='relu'))
	# # model.add(Dropout(0.2))
	# # model.add(Dense(units=1024, activation='relu'))
	# # model.add(Dropout(0.2))
	# # model.add(Dense(units=1024, activation='relu'))
	# # model.add(Dropout(0.2))
	# # model.add(Dense(1, activation='sigmoid'))

	# # model = cnn_rnn(x_train.shape[1:])
	# model.add(LSTM(1024, return_sequences=True, input_shape=x_train.shape[1:]))
	# model.add(LSTM(1024, return_sequences=False))
	# model.add(Dense(512,activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(1,activation='sigmoid'))

	# epochs = 5
	# batch_size = 32

	# model.compile(optimizer='adam',
 #                  loss=losses.binary_crossentropy,
 #                  metrics=['accuracy'])
	# model.summary()

	# training_start = time()
	# print("starting training")

	# H = model.fit(x_train, y_train,
	# 			  batch_size=batch_size,
	# 			  epochs=epochs,
	# 			  verbose=1,
	# 			  shuffle=True,
	# 			  validation_data=(x_val, y_val))

	# training_time = time() - training_start
	# print("training finished")
	# model.summary()

	#SVM
	from sklearn import svm
	x_train = x_train.reshape(x_train.shape[0], -1)
	x_val = x_val.reshape(x_val.shape[0],-1)
	model = svm.SVC(class_weight='balanced')
	model.fit(x_train, y_train)

	# inference_start = time()
	# loss, accuracy = model.evaluate(x=x_val, y=y_val)
	# print(loss, accuracy)
	from sklearn.metrics import confusion_matrix
	# y_pred = model.predict_classes(x=x_val)
	y_pred = model.predict(x_val).flatten()
	y_pred = np.round(y_pred).astype(np.uint8)
	con_mat = confusion_matrix(y_val,y_pred)

	class_names = ['no bluffing','bluffing']

	plt.figure()
	plot_confusion_matrix(con_mat, classes=class_names,
						  title='Confusion matrix, without normalization')

	plt.figure()
	plot_confusion_matrix(con_mat, classes=class_names, normalize=True,
						  title='Normalized confusion matrix')

	plt.show()
	# inference_time = time() - inference_start
