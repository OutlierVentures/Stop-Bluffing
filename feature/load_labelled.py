import csv
import pandas

from random import randint
import numpy as np

FRAMES_PER_CLIP = 150
DATA_PATH = 'data/labelled_au.csv'

def loso_splitter(loso, player_out, au_type, filename):
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

def read_labelled(player_out=0, au_type="(r|c)", filename=DATA_PATH):
	labelled = pandas.read_csv(filename, sep='\s*,\s*', engine='python')

	#If training set, the rest players stays
	loso = labelled[labelled.playerId != player_out]
	loso = loso.reset_index(drop=True)
	sample_training, isBluff_training = loso_splitter(loso, player_out, au_type, filename)

	#If testing set, one player stays
	loso = labelled[labelled.playerId == player_out]
	loso = loso.reset_index(drop=True)
	sample_testing, isBluff_testing = loso_splitter(loso, player_out, au_type, filename)

	return np.asarray(sample_training), np.asarray(isBluff_training), np.asarray(sample_testing), np.asarray(isBluff_testing)

def split_dataset(x_dataset, y_dataset, ratio):
	split_idx = int(ratio * len(x_dataset))
	x_split0 = np.array(x_dataset[:split_idx])
	y_split0 = np.array(y_dataset[:split_idx])
	x_split1 = np.array(x_dataset[split_idx:])
	y_split1 = np.array(y_dataset[split_idx:])
	return x_split0, y_split0, x_split1, y_split1
