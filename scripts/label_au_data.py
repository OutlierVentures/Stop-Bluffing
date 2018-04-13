import pandas
import os
# import progressbar

FRAMES = 150
IGNORE_FRAMES = [38,63,74]
BLUFF_DATA_PATH = 'data/bluff_data.csv'
AU_DATA_PATH = 'data/au_data.csv'
AU_OUT_PATH = 'data/labelled_au.csv'

def remove_columns(data, col_names):
    for col in col_names:
        data = data.drop(col, axis=1)
    return data

def add_labels(data, labels, players):
    # initialise column for isBluffing label
    data['isBluffing'] = None
    data['clipId'] = None
    data['playerId'] = None

    # set up progress bar
    bar = progressbar.ProgressBar(maxval=data.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    # loop through each label (each clip label)
    count = 0
    for idx, label in enumerate(labels):
        if (idx+1) in IGNORE_FRAMES:
            continue
        else:
            for i in range(FRAMES):
                bar.update((FRAMES*count)+i)

                # stop when reaches end of data
                if (FRAMES*count)+i == data.shape[0]:
                    return data

                # assign label
                data.ix[(FRAMES*count)+i, 'isBluffing'] = label
                data.ix[(FRAMES*count)+i, 'clipId'] = idx+1
                data.ix[(FRAMES*count)+i, 'playerId'] = players[idx]
            count += 1
    bar.finish()
    return data

def remove_low_confidence(data):

    low_confidence_frames = []

    count = 1
    total_confidence = 0

    for idx, row in data.iterrows():
        if count == 150:
            avg_confidence = total_confidence/150
            if avg_confidence < 0.90:
                data = data.drop(data.index[idx-150:idx])

            total_confidence = 0
            count = 1
        else:
            total_confidence += row[' confidence']
            count += 1
    return data

if __name__ == '__main__':
    # get isBluffing labels
    annotated = pandas.read_csv(BLUFF_DATA_PATH)
    labels = annotated['isBluffing']
    players = annotated['playerId']

    # load au data -> remove columns that are not needed -> add labels
    au_data = pandas.read_csv(AU_DATA_PATH)

    au_data = remove_columns(au_data, ['frame',' face_id',' timestamp',' success'])
    au_data = add_labels(au_data, labels, players)
    au_data = remove_low_confidence(au_data)

    # save labelled data
    au_data.to_csv(AU_OUT_PATH, sep=',', index=False)
