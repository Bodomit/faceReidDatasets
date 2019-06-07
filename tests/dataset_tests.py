import unittest
import faceReidDatasets


class DatasetBaseBasic(unittest.TestCase):
    def setUp(self):
        self.data = [
            ("A", 1),
            ("B", 2),
            ("C", 3),
            ("D", 1),
            ("E", 2),
        ]

    def test_init(self):
        dataset = faceReidDatasets.datasets.DatasetBase(self.data)
        self.assertIsNotNone(dataset)
