import csv
import os
import cv2
import subprocess

def process_info(csv_info):
    # get video filepath
    filepath = get_filepath(csv_info)

    # check file exists
    if os.path.exists(filepath):
        extract_frames_and_make_vid(filepath,csv_info)


def get_filepath(csv_info):
    filename = 'G' + csv_info[1] + 'R' + csv_info[2]
    filepath = 'data/videos/' + csv_info[3] + '/' + filename
    if os.path.exists(filepath + '.mov'):
        filepath = filepath + '.mov'
    else:
        filepath = filepath + '.mp4'
    return filepath

def extract_frames_and_make_vid(filepath, csv_info):
    clip_id = csv_info[0]
    start_frame = int(csv_info[4])
    end_frame = start_frame + 149
    output_folder = 'data/frames/' + clip_id + '/'

    # make output folder for frames if doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_cap = cv2.VideoCapture(filepath)

    # iterate through all frames
    count = 0
    success,image = vid_cap.read()
    while success:
        success,image = vid_cap.read()
        # save frame if between start and end frame
        if count >= start_frame and count <= end_frame:
            cv2.imwrite(output_folder + str(count) + '.jpg', image)
        elif count > end_frame:
            break;
        count += 1

    # save video from extracted frames folder
    make_vid(output_folder, clip_id, start_frame)

def make_vid(imgs_folder, clip_id, start_frame):
    # create output video folder
    output_video_folder = 'data/trimmed/'
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    # set output video name
    video_filepath = output_video_folder + clip_id

    # save video from iamges
    subprocess.call(
        'ffmpeg -pattern_type glob -framerate 30 -i "%s*.jpg" %s.mp4' % (imgs_folder,video_filepath),
        shell=True)


if __name__ == '__main__':
    with open('bluff_data.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            process_info(row)
