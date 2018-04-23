# Stop-Bluffing

Machine Learning model for detecting bluffing in a Poker game.

A mini project part of UCL's Affective Computing and Human Robot Interaction course. 

## Prepare Data

Place videos in `data/videos/{playerId}`. Video filenames should have the format `G{gameId}R{roundId}`.

E.g. the video of player 2 in game 1 round 2 would be located at `data/videos/2/G1R2.mov`.

## Facial Action Unit Model

The Facial Action Coding System (Ekman et al., 2002) breaks facial geometry into smaller parts and encodes how these change over time ('Action Units' (AUs)).
An AU is movement of a muscle or group of muscles. For example, AU10 is the raising of the upper lip.
AUs are scored by intensity and can encode almost any anatomically possible facial expression.

Full guide by Ekman et al.: https://bit.do/facs-full

Summary: https://bit.do/facs-summary

### Extracting action units

Extraction is done using Openface (Baltrušaitis et al., 2016).

We recommend using Windows. We had success with MacOS and Ubuntu but configuring components is considerably more challenging.
For MacOS see https://bit.do/openface-mac and for other UNIX systems see https://bit.do/openface-unix.

#### Instructions for Windows

*Requires Visual Studio 2015 **without updates** (https://bit.do/openface-vs) and Visual C++ Redistributable 2015 (https://bit.do/openface-cpp).*

Download the required binaries with this link: http://bit.do/openface-bins

Clone OpenFace into your repository:
```
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
```

Open OpenFace.sln with Visual Studio 2015 and run OpenFaceOffline.
In the menu, set recorded data to AUs only.
Open your selected files and run OpenFace.

Outputs are saved as a CSV.
In the header, a suffix of *c* denotes classification data and a suffix of *r* denotes regression data.

### Training

To train the SVM model:
```
python -m feature.svm
```

To train the neural network models:
```
python -m feature.keras_models [model]
```

The [model] argument can be 'mlp' or 'rnn', for example:
```
python -m feature.keras_models mlp
```

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
