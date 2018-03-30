import csv
import unittest
import numpy as np
from model import loader


class TestLoader(unittest.TestCase):
    def test_load(self):
        x, y = loader.load(shuffle=False)

        nb_x = x.shape[0]
        nb_y = len(y)

        # Test that number of samples match
        self.assertEqual(nb_x, nb_y)

        # Test that labels match csv
        with open('data/bluff_data.csv') as fp:
            reader = csv.DictReader(fp)
            labels = []
            for row in reader:
                if row['error'] != '':
                    continue
                labels.append(float(row['isBluffing']))

        np.testing.assert_array_equal(y, np.array(labels))

    def test_compact_frames(self):
        x = np.array([
            [  # Sample 1
                [  # Time 1
                    [1, 1, 1],  # Face landmark 1
                    [2, 2, 2]
                ],
                [  # Time 2
                    [1, 1, 1],  # Face landmark 1
                    [2, 2, 2]
                ],
                [  # Time 3
                    [1, 1, 1],  # Face landmark 1
                    [2, 2, 2]
                ],
                [  # Time 4
                    [1, 1, 1],  # Face landmark 1
                    [2, 2, 2]
                ],
                [  # Time 5
                    [1, 1, 1],  # Face landmark 1
                    [2, 2, 2]
                ],
                [  # Time 6
                    [2, 2, 2],  # Face landmark 1
                    [3, 3, 3]
                ],
                [  # Time 7
                    [3, 3, 3],  # Face landmark 1
                    [4, 4, 4]
                ],
            ]
        ])
        compacted = loader.compact_frames(x)

        self.assertEqual(compacted.shape, (1, 2, 2, 3))
