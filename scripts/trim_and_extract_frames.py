import csv
import os
import subprocess
import progressbar
import cv2

DATA_DIR = '../data'


def process_clip(row):
    # get video file_path
    file_path = get_file_path(row)

    # check file exists
    if os.path.exists(file_path):
        extract_frames_and_make_vid(file_path, row)
    else:
        raise FileNotFoundError(file_path)


def get_file_path(csv_info):
    filename = 'G' + csv_info[1] + 'R' + csv_info[2]
    file_path = os.path.join(DATA_DIR, 'videos', csv_info[3], filename)
    if os.path.exists(file_path + '.mov'):
        file_path = file_path + '.mov'
    else:
        file_path = file_path + '.mp4'
    return file_path


def extract_frames_and_make_vid(file_path, csv_info):
    clip_id = csv_info[0]
    start_frame = int(csv_info[4])
    end_frame = start_frame + 149
    output_folder = os.path.join(DATA_DIR, 'frames', clip_id)

    # make output folder for frames if doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_cap = cv2.VideoCapture(file_path)

    # iterate through all frames
    count = 0
    success, image = vid_cap.read()
    while success:
        success, image = vid_cap.read()
        # save frame if between start and end frame
        if start_frame <= count <= end_frame:
            cv2.imwrite(os.path.join(output_folder, "{:05d}.jpg".format(count)), image)
        elif count > end_frame:
            break
        count += 1

    # save video from extracted frames folder
    make_vid(output_folder, clip_id)


def make_vid(imgs_folder, clip_id):
    devnull = open(os.devnull, 'w')

    # create output video folder
    output_video_folder = os.path.join(DATA_DIR, 'trimmed')
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    # set output video name
    video_file_path = os.path.join(output_video_folder, clip_id)

    # save video from images
    subprocess.call(
        'ffmpeg -pattern_type glob -framerate 30 -i "{}/*.jpg" {}.mp4'.format(imgs_folder, video_file_path),
        shell=True,
        stdout=devnull)


if __name__ == '__main__':
    with open('../data/bluff_data.csv') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader) # skip header row
        bar = progressbar.ProgressBar()
        for row in bar(reader):
            process_clip(row)
