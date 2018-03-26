import unittest
import numpy as np
from model import loader


class TestLoader(unittest.TestCase):
    def test_load(self):
        x, y = loader.load()

        nb_x = x.shape[0]
        nb_y = len(y)

        # Test that number of samples match
        self.assertEqual(nb_x, nb_y)

        # Type of y must be uint8 (binary labels)s
        self.assertEqual(y.dtype, np.uint8)
