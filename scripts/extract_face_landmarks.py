"""
This script should be run with CUDA
"""
import glob
import os
import time
import json
import face_alignment

FRAMES_DIR = '../data/frames'
OUTPUT_DIR = '../data/face-landmarks'

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)


def process_folder(dir, all_faces=False):
    types = ('*.jpg', '*.png')
    images_list = []
    for files in types:
        images_list.extend(glob.glob(os.path.join(dir, files)))

    images_list.sort()

    predictions = []
    for image_name in images_list:
        face_preds = fa.get_landmarks(image_name, all_faces)

        # No face detected
        if face_preds is None:
            return None

        predictions.append((image_name, face_preds[-1].tolist()))

    return predictions


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for file in os.listdir(FRAMES_DIR):
    path = os.path.join(FRAMES_DIR, file)
    if os.path.isdir(path):
        clip_id = os.path.basename(file)
        print("Processing clip {}".format(clip_id))
        output_path = os.path.join(OUTPUT_DIR, "{}.json".format(clip_id))

        if os.path.exists(output_path):
            print('Skipping {}'.format(clip_id))
            continue

        start = time.time()
        landmarks = process_folder(path, all_faces=False)

        # Contained a frame where face was not detected
        if landmarks is None:
            print('No face detected, skipping {}'.format(clip_id))
            continue

        with open(output_path, 'w') as f:
            json.dump(landmarks, f)

        print("Took {:.2f}s".format(time.time() - start))
