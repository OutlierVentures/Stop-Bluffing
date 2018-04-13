"""
Keras Models
"""
import argparse
from collections import Counter
from keras import losses
from keras import callbacks
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from feature import load_labelled
from model import evaluate

def mlp(input_shape):
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	return model

def rnn(input_shape):
	model = Sequential()
	model.add(LSTM(1024, return_sequences=True, input_shape=input_shape))
	model.add(LSTM(1024, return_sequences=False))
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1,activation='sigmoid'))
	return model

def get_model(name, input_shape):
	""""
	Retrieve model architecture depending on cmd args
	"""
	if name == 'mlp':
		return mlp(input_shape)
	if name == 'rnn':
		return rnn(input_shape)
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
	x_train, y_train, x_val, y_val = load_labelled.read_labelled(1)

	model = get_model(args.model, input_shape=x_train.shape[1:])

	model.compile(optimizer='adam',
				  loss=losses.binary_crossentropy,
				  metrics=['accuracy'])
	model.summary()

	checkpointer = callbacks.ModelCheckpoint(filepath="data/feature_weights.hdf5", verbose=1, save_best_only=True)
	# early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2)

	class_weights = get_class_weights(y_train)
	model.fit(x_train, y_train,
			  epochs=50,
			  batch_size=8,
			  validation_data=(x_val, y_val),
			  callbacks=[
				  checkpointer
				  # early_stopping,
			  ],
			  class_weight=class_weights,
			  )

	# Load best model
	model.load_weights('data/feature_weights.hdf5')
	
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