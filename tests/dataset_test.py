import os
import unittest
import tempfile
import pickle

from freidds import datasets


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

    def test_as_column_lists(self):
        dataset = datasets.DatasetBase(self.data).as_column_lists()
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset[0]), len(dataset[1]))


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

        dataset2 = datasets.VGGFace2(self.dataset_directory,
                                     cache_directory=cache_dir)

        # Check paths are correct.
        self.assertTrue(os.path.exists(dataset2["train"][0][0]))
        self.assertTrue(os.path.exists(dataset2["train"][1][0]))
        self.assertTrue(os.path.exists(dataset2["train"][2][0]))


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

    def test_filters(self):
        filters = {
            "lighting": ["A"],
            "zenith": [90],
            "azimuth": [270],
            "resolution": None
        }
        dataset = datasets.Synthetic(self.dataset_directory, filters=filters)
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset["train"]), 6576)

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

        dataset2 = datasets.Synthetic(self.dataset_directory,
                                      cache_directory=cache_dir)

        # Check paths are correct.
        self.assertTrue(os.path.exists(dataset2["train"][0][0]))
        self.assertTrue(os.path.exists(dataset2["train"][1][0]))
        self.assertTrue(os.path.exists(dataset2["train"][2][0]))

    def test_init_cache_read_with_filters(self):
        filters = {
            "lighting": ["A"],
            "zenith": [90],
            "azimuth": [270],
            "resolution": None
        }
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "synth.pickle")
        dataset = datasets.Synthetic(self.dataset_directory,
                                     cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

        dataset2 = datasets.Synthetic(self.dataset_directory,
                                      filters=filters,
                                      cache_directory=cache_dir)

        self.assertEqual(len(dataset["train"]), 736488)
        self.assertEqual(len(dataset2["train"]), 6576)

    def test_init_cache_read_with_filters2(self):
        filters = {
            "lighting": ["A"],
            "zenith": [90],
            "azimuth": [270],
            "resolution": None
        }
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "synth.pickle")
        dataset = datasets.Synthetic(self.dataset_directory,
                                     filters=filters,
                                     cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

        dataset2 = datasets.Synthetic(self.dataset_directory,
                                      cache_directory=cache_dir)

        self.assertEqual(len(dataset["train"]), 6576)
        self.assertEqual(len(dataset2["train"]), 736488)


class COXFaceDBTests(unittest.TestCase):
    def setUp(self):
        self.dataset_directory = os.path.join(
            "~",
            "datasets",
            "coxfacedb"
        )

    def test_init(self):
        dataset = datasets.COXFaceDB(self.dataset_directory)
        self.assertIsNotNone(dataset)

        # Test for 4 subsets.
        self.assertSetEqual(set(dataset), set(["still",
                                               "cam1",
                                               "cam2",
                                               "cam3"]))

        # Check classes are in correct sub directory.
        self.assertTrue("201103180001" in (x[1] for x in dataset["still"]))
        self.assertTrue("201104240300" in (x[1] for x in dataset["cam3"]))

        # Check all files exist.
        self.assertTrue(all(os.path.isfile(x[0]) for x in dataset["still"]))
        self.assertTrue(all(os.path.isfile(x[0]) for x in dataset["cam1"]))
        self.assertTrue(all(os.path.isfile(x[0]) for x in dataset["cam2"]))
        self.assertTrue(all(os.path.isfile(x[0]) for x in dataset["cam3"]))

    def test_v2s_dataset(self):
        dataset = datasets.COXFaceDB(self.dataset_directory)
        dataset = dataset.get_v2s(gallery_and_probe=False)

        # Test for two subsets.
        self.assertSetEqual(set(dataset), set(["train", "test"]))

        # Test for galleries and probes.
        correct = set(["still", "cam1", "cam2", "cam3"])
        self.assertSetEqual(set(dataset["train"][0].keys()), correct)
        self.assertSetEqual(set(dataset["train"][9].keys()), correct)
        self.assertSetEqual(set(dataset["test"][0].keys()), correct)
        self.assertSetEqual(set(dataset["test"][9].keys()), correct)

    def test_v2s_dataset_gallery_and_probe(self):
        dataset = datasets.COXFaceDB(self.dataset_directory)
        dataset = dataset.get_v2s(gallery_and_probe=True)

        # Test for two subsets.
        self.assertSetEqual(set(dataset), set(["train", "test"]))

        # Test for galleries and probes.
        correct = set(["gallery", "probe"])
        self.assertSetEqual(set(dataset["train"][0].keys()), correct)
        self.assertSetEqual(set(dataset["train"][9].keys()), correct)
        self.assertSetEqual(set(dataset["test"][0].keys()), correct)
        self.assertSetEqual(set(dataset["test"][9].keys()), correct)

    def test_init_cache_created(self):
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "coxfacedb.pickle")
        dataset = datasets.COXFaceDB(self.dataset_directory,
                                     cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

    def test_init_cache_read(self):
        cache_dir = tempfile.mkdtemp()
        cache_path = os.path.join(cache_dir, "coxfacedb.pickle")
        dataset = datasets.COXFaceDB(self.dataset_directory,
                                     cache_directory=cache_dir)
        self.assertIsNotNone(dataset)
        self.assertTrue(os.path.exists(cache_path))

        dataset2 = datasets.COXFaceDB(self.dataset_directory,
                                      cache_directory=cache_dir)

        # Check paths are correct.
        self.assertTrue(os.path.exists(dataset2["still"][0][0]))
        self.assertTrue(os.path.exists(dataset2["cam1"][0][0]))
        self.assertTrue(os.path.exists(dataset2["cam2"][-1][0]))
        self.assertTrue(os.path.exists(dataset2["cam3"][-1][0]))


class MMFTests(unittest.TestCase):
    def setUp(self):
        self.dataset_directory = os.path.join(
            "~",
            "datasets",
            "mmf"
        )

    def test_init(self):
        dataset = datasets.MMF(self.dataset_directory)
        self.assertIsNotNone(dataset)

        # Test for 2 subsets.
        self.assertSetEqual(set(dataset), set(["A", "B"]))

        # Check classes are in correct sub directory.
        self.assertTrue("001" in (x[1] for x in dataset["A"]))
        self.assertTrue("001" in (x[1] for x in dataset["B"]))

        # Check all files exist.
        self.assertTrue(all(os.path.isfile(x[0]) for x in dataset["A"]))
        self.assertTrue(all(os.path.isfile(x[0]) for x in dataset["B"]))

    def test_ignore_filters_kwarg(self):
        dataset = datasets.MMF(self.dataset_directory, filters=None)
        self.assertIsNotNone(dataset)


class CVMultiViewWrapperTestsWithMMF(unittest.TestCase):
    def setUp(self):
        self.dataset_directory = os.path.join(
            "~",
            "datasets",
            "mmf"
        )

    def test_init(self):
        dataset = datasets.MMF(self.dataset_directory)
        cv_dataset = datasets.CVMultiViewWrapper(dataset, 11, "A")
        self.assertIsNotNone(cv_dataset)

        # Test for 11 rounds.
        self.assertEqual(len(cv_dataset), 11)

        # Check for test and train subsets in each round.
        for round in cv_dataset:
            self.assertIn("train", round)
            self.assertIn("test", round)

        # Check all views in all rounds in all subsets.
        for round in cv_dataset:
            for subset in round:
                self.assertIn("A", round[subset])
                self.assertIn("B", round[subset])

    def test_v2s_dataset_gallery_and_probe(self):
        dataset = datasets.MMF(self.dataset_directory)
        cv_dataset = datasets.CVMultiViewWrapper(dataset, 11, "A")
        cv_dataset_v2s = cv_dataset.get_v2s(gallery_and_probe=True)

        # Test for two subsets.
        self.assertEqual(len(cv_dataset_v2s), 11)

        # Test for galleries and probes.
        correct = set(["gallery", "probe"])
        for i in range(11):
            self.assertSetEqual(
                set(cv_dataset_v2s[i]["train"].keys()), correct)
            self.assertSetEqual(set(cv_dataset_v2s[i]["test"].keys()), correct)

        # Test galleries are unique.
        for i in range(11):
            test_gallery = cv_dataset_v2s[i]["test"]["gallery"]
            self.assertEqual(len(test_gallery), len(set(test_gallery)))

            train_gallery = cv_dataset_v2s[i]["train"]["gallery"]
            self.assertEqual(len(train_gallery), len(set(train_gallery)))
