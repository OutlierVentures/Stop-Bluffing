# Stop-Bluffing
Machine Learning model for detecting bluffs in Poker

## Face Alignment

Use [face-alignment](https://github.com/1adrianb/face-alignment) library to extract 3D face landmarks.

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

cd /stop-bluffing
python test/test_face_alignment.py
```