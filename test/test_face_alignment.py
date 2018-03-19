import argparse
import glob
import os
import face_alignment
import time

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
args = parser.parse_args()

print("Using cuda: {}".format(args.cuda))


def process_folder(path, all_faces=False):
    types = ('*.jpg', '*.png')
    images_list = []
    for files in types:
        images_list.extend(glob.glob(os.path.join(path, files)))

    predictions = []
    for image_name in images_list:
        predictions.append(
            (image_name, fa.get_landmarks(image_name, all_faces)))

    return predictions


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=args.cuda, flip_input=False)

start = time.time()
preds = process_folder('assets/', all_faces=True)
print("Took {}s".format(time.time() - start))
