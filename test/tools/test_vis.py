import unittest
import matplotlib.pyplot as plt

from model import loader
from tools import vis


class TestVis(unittest.TestCase):
    def setUp(self):
        self.x, self.y = loader.load(shuffle=False)

    def test_vis_many_landmarks(self):
        vis.vis_many_face_landmarks(self.x[0, :, :, :])

    def test_vis_landmarks(self):
        vis.vis_face_landmarks(self.x[0, 0, :, :])
        plt.show()
