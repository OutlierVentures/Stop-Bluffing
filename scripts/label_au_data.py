import pandas
import os
import progressbar

FRAMES = 150
IGNORE_FRAMES = [38,63,74]

def remove_columns(data, col_names):
    for col in col_names:
        data = data.drop(col, axis=1)
    return data

def add_labels(data, labels):
    # initialise column for isBluffing label
    data['isBluffing'] = None
    data['clip_id'] = None

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
                data.ix[(FRAMES*count)+i, 'clip_id'] = idx+1
            count += 1
    bar.finish()
    return data

if __name__ == '__main__':
    # get isBluffing labels
    annotated = pandas.read_csv('annotated_data.csv')
    labels = annotated['isBluffing']

    # load au data -> remove columns that are not needed -> add labels
    au_data = pandas.read_csv('frames.csv')
    au_data = remove_columns(au_data, ['frame',' face_id',' timestamp',' success'])
    au_data = add_labels(au_data, labels)

    # save labelled data
    au_data.to_csv('labelled.csv', sep=',', index=False)
