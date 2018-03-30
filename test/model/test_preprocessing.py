import unittest
import numpy as np

from model import loader, preprocessing


class TestPreprocessing(unittest.TestCase):
    def test_fisher_vector(self):
        x, y = loader.load(shuffle=False)
        fv = preprocessing.to_fisher(x)