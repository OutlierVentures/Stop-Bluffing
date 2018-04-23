# Stop-Bluffing

Machine Learning model for detecting bluffing in a Poker game.

A mini project part of UCL's Affective Computing and Human Robot Interaction course. 

## Prepare Data

Place videos in `data/videos/{playerId}`. Video filenames should have the format `G{gameId}R{roundId}`.

E.g. the video of player 2 in game 1 round 2 would be located at `data/videos/2/G1R2.mov`.

## Facial Action Unit Model

TODO:

### Extracting action units

### Training

## Facial Geometry Model

Our other model attempted to use only the coordinates of the subject's facial geometry.

### Prerequisites

* Nvidia Docker

```
# Build docker image
docker build -t face-alignment .

# Launch Docker container
sudo nvidia-docker run -it \
  -v ~/Stop-Bluffing:/stop-bluffing \
  --rm \
  face-alignment

python test/test_face_alignment.py
```

### Trim and extract frames

Run `python scripts/trim_and_extract_frames.py`. This will extract the 5s clips as an image sequence and place them in `data/frames`.

### Extract facial geometry

Use [face-alignment](https://github.com/1adrianb/face-alignment) library to extract 3D face landmarks.

You will need a GPU to run this. 

```
python scripts/extract_face_landmarks.py
```

### Train model

After all the necessary data pre-processing is completed, you can now train the model.

```
python train.py [model]
```

For example, to train the MLP model:

```
python train.py mlp
```

It will automatically evaluate on the test set and show a confusion matrix with Matplotlib.