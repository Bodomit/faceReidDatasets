import os
import unittest
import tempfile
import pickle

from faceReidDatasets import datasets


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
        dataset = datasets.DatasetBase(self.data)
        self.assertIsNotNone(dataset)
        sorted_dataset = [("A", 1),
                          ("D", 1),
                          ("B", 2),
                          ("E", 2),
                          ("C", 3)]
        self.assertListEqual(dataset.dataset, sorted_dataset)

    def test_iterator(self):
        dataset = datasets.DatasetBase(self.data)
        sorted_dataset = [("A", 1),
                          ("D", 1),
                          ("B", 2),
                          ("E", 2),
                          ("C", 3)]
        self.assertListEqual(list(dataset), sorted_dataset)

    def test_as_target_to_source_list(self):
        dataset = datasets.DatasetBase(self.data).as_target_to_source_list()
        correct = {1: ["A", "D"],
                   2: ["B", "E"],
                   3: ["C"]}
        self.assertDictEqual(dataset, correct)


class MultiLevelDatasetBaseBasic(unittest.TestCase):
    def setUp(self):
        self.data = {
            "train": [("A", 1),
                      ("B", 2),
                      ("C", 3),
                      ("D", 1),
                      ("E", 2)],
            "test":  [("F", 10),
                      ("G", 11),
                      ("H", 10)]
        }

    def test_init(self):
        dataset = datasets.MutiLevelDatasetBase(self.data)
        self.assertIsNotNone(dataset)
        sorted_dataset = {
            "train": [("A", 1),
                      ("D", 1),
                      ("B", 2),
                      ("E", 2),
                      ("C", 3)],
            "test":  [("F", 10),
                      ("H", 10),
                      ("G", 11)]}
        self.assertListEqual(dataset.dataset["train"].dataset,
                             sorted_dataset["train"])
        self.assertListEqual(dataset.dataset["test"].dataset,
                             sorted_dataset["test"])

    def test_iterator(self):
        dataset = datasets.MutiLevelDatasetBase(self.data)
        self.assertListEqual(sorted(list(dataset)), sorted(["train", "test"]))

    def test_as_target_to_source_list(self):
        dataset = datasets.MutiLevelDatasetBase(self.data)
        dataset = dataset.as_target_to_source_list()
        correct = {
            "train": {1: ["A", "D"],
                      2: ["B", "E"],
                      3: ["C"]},

            "test":  {10: ["F", "H"],
                      11: ["G"]}
        }
        self.assertDictEqual(dataset, correct)


class VGGFace2Tests(unittest.TestCase):
    def setUp(self):
        self.dataset_directory = os.path.join(
            "~",
            "datasets",
            "vggface2"
        )

    def test_init(self):
        dataset = datasets.VGGFace2(self.dataset_directory)
        self.assertIsNotNone(dataset)

        # Test for two subsets.
        self.assertSetEqual(set(dataset), set(["train", "test"]))

        # Check classes are in correct sub directory.
        self.assertTrue("n000002" in (x[1] for x in dataset["train"]))
        self.assertTrue("n000001" in (x[1] for x in dataset["test"]))

    def test_v2s_dataset(self):
        dataset = datasets.VGGFace2(self.dataset_directory).get_v2s()

        # Test for two subsets.
        self.assertSetEqual(set(dataset), set(["train", "test"]))

        # Test for galleries and probes.
        self.assertSetEqual(set(dataset["train"]), set(["gallery", "probe"]))
        self.assertSetEqual(set(dataset["test"]), set(["gallery", "probe"]))

        # Test length of galleries.
        self.assertEqual(len(dataset["train"]["gallery"]), 8631)
        self.assertEqual(len(dataset["test"]["gallery"]), 500)

    def test_init_cache_created(self):
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "vggface2.pickle")
        dataset = datasets.VGGFace2(self.dataset_directory,
                                    cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

    def test_init_cache_read(self):
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "vggface2.pickle")
        dataset = datasets.VGGFace2(self.dataset_directory,
                                    cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

        # Modfy dataset and cache manually.
        dataset.dataset["test"] = []
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset.dataset, f)

        dataset2 = datasets.VGGFace2(self.dataset_directory,
                                     cache_directory=cache_dir)
        self.assertEqual(len(dataset2.dataset["test"]), 0)


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.dataset_directory = os.path.join(
            "~",
            "datasets",
            "synth"
        )

    def test_init(self):
        dataset = datasets.Synthetic(self.dataset_directory)
        self.assertIsNotNone(dataset)

        # Test for two subsets.
        self.assertSetEqual(set(dataset), set(["train", "test"]))

        # Check classes are in correct sub directory.
        self.assertTrue("00001" in (x[1] for x in dataset["train"]))
        self.assertTrue("00052" in (x[1] for x in dataset["test"]))

    def test_v2s_dataset(self):
        dataset = datasets.Synthetic(self.dataset_directory).get_v2s()

        # Test for two subsets.
        self.assertSetEqual(set(dataset), set(["train", "test"]))

        # Test for galleries and probes.
        self.assertSetEqual(set(dataset["train"]), set(["gallery", "probe"]))
        self.assertSetEqual(set(dataset["test"]), set(["gallery", "probe"]))

        # Test length of galleries.
        self.assertEqual(len(dataset["train"]["gallery"]), 1644)
        self.assertEqual(len(dataset["test"]["gallery"]), 100)

    def test_init_cache_created(self):
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "synth.pickle")
        dataset = datasets.Synthetic(self.dataset_directory,
                                     cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

    def test_init_cache_read(self):
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "synth.pickle")
        dataset = datasets.Synthetic(self.dataset_directory,
                                     cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

        # Modfy dataset and cache manually.
        dataset.dataset["test"] = []
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset.dataset, f)

        dataset2 = datasets.Synthetic(self.dataset_directory,
                                      cache_directory=cache_dir)
        self.assertEqual(len(dataset2.dataset["test"]), 0)