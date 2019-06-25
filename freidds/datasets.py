import os
import re
import abc
import random
import itertools
import sacred
import pickle
import collections

from typing import List, Set, Tuple

ingrediant = sacred.Ingredient("dataset")


class DatasetBase(collections.abc.Sequence):
    """
    #TODO
    """

    def __init__(self, dataset: List[Tuple[str, str]]):
        self._dataset = sorted(dataset, key=lambda x: (x[1], x[0]))

    @property
    def dataset(self):
        return self._dataset

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __len__(self):
        return self.dataset.__len__()

    def as_target_to_source_list(self):
        target_to_source_list = {}
        for k, g in itertools.groupby(self._dataset, lambda x: x[1]):
            target_to_source_list[k] = [i[0] for i in g]
        return target_to_source_list

    def as_column_lists(self):
        return list(zip(*self.dataset))


class MutiLevelDatasetBase(abc.ABC):
    """
    #TODO
    """

    def __init__(self, dataset: dict):
        self._dataset = self._populate_dataset(dataset)

    def _traverse(self, node, f):
        if isinstance(node, list) or isinstance(node, DatasetBase):
            return f(node)
        elif isinstance(node, dict):
            return {k: self._traverse(v, f) for k, v in node.items()}
        else:
            raise ValueError("Node must be a list of tuples.")

    def _populate_dataset(self, dataset):
        return self._traverse(dataset, lambda d: DatasetBase(d))

    @property
    def dataset(self):
        return self._dataset

    def __iter__(self):
        return self.dataset.__iter__()

    def __getitem__(self, key):
        return self.dataset[key]

    def as_target_to_source_list(self):
        return self._traverse(self.dataset,
                              lambda d: d.as_target_to_source_list())

    @staticmethod
    def _read_train_test_dataset(dataset_directory, image_ext):
        if not os.path.isdir(dataset_directory):
            raise NotADirectoryError()

        dataset = {
            "train": [],
            "test": []
        }

        for root, dirs, files in os.walk(dataset_directory, topdown=True):
            assert not (dirs and files)

            for file in files:
                if os.path.splitext(file)[1] == image_ext:
                    path = os.path.join(root, file)
                    path = os.path.expanduser(path)
                    path = os.path.abspath(path)
                    label = os.path.basename(os.path.dirname(path))
                    subset = os.path.basename(os.path.dirname(root))
                    dataset[subset].append((path, label))

        return dataset


class ReadableMultiLevelDatasetBase(MutiLevelDatasetBase, abc.ABC):
    """
    #TODO
    """
    def __init__(self, dataset_directory, **kwargs):
        dataset_directory = os.path.expanduser(dataset_directory)
        dataset_directory = os.path.abspath(dataset_directory)
        self.dataset_directory = dataset_directory
        super().__init__(
            self._read_dataset_via_cache(dataset_directory, **kwargs)
        )

    def _read_dataset_via_cache(self,
                                dataset_directory,
                                cache_directory=None):
        dataset_name = os.path.basename(dataset_directory)
        if cache_directory is None:
            return self._read_dataset()

        # Read / Store in the cache.
        try:
            cache_directory = os.path.expanduser(cache_directory)
            cache_directory = os.path.abspath(cache_directory)
            cache_path = os.path.join(cache_directory,
                                      dataset_name + ".pickle")
            with open(cache_path, "rb") as f:
                # Prefix dataset directory.
                scrubbed_dataset = pickle.load(f)
                return self._prefix_dataset_directory(scrubbed_dataset)

        except FileNotFoundError:
            dataset = self._read_dataset()
            try:
                # Scrub dataset of dataset_directory.
                scrubbed_dataset = self._scrub_dataset_directory(dataset)
                with open(cache_path, 'wb') as f:
                    pickle.dump(scrubbed_dataset, f)
            finally:
                return dataset

    @abc.abstractmethod
    def _read_dataset(self):
        pass

    def _scrub_dataset_directory(self, dataset):

        def scrub(dataset):
            scrubbed_samples = []
            for sample in dataset:
                scrubbed_samples.append(
                    (sample[0].replace(self.dataset_directory, ""), sample[1]))
            return scrubbed_samples

        return self._traverse(dataset, scrub)

    def _prefix_dataset_directory(self, dataset):

        def prefix(dataset):
            prefixed_samples = []
            for sample in dataset:
                prefixed_samples.append(
                    (
                        os.path.join(self.dataset_directory, sample[0]),
                        sample[1]
                    )
                )
            return prefixed_samples

        return self._traverse(dataset, prefix)


class VGGFace2(ReadableMultiLevelDatasetBase):
    """
    #TODO
    """

    def _read_dataset(self):
        return self._read_train_test_dataset(self.dataset_directory, ".jpg")

    def get_v2s(self, seed=42):
        random.seed(seed)
        return {
            "train": self._v2s_subset(self.dataset["train"]),
            "test": self._v2s_subset(self.dataset["test"])
        }

    def _v2s_subset(self, subset):
        gallery = []
        probe = []
        for label, paths in subset.as_target_to_source_list().items():
            gallery_candidate = paths.pop(random.randrange(0, len(paths)))
            gallery.append((label, gallery_candidate))
            probe.extend((label, p) for p in paths)
        return {"gallery": gallery, "probe": probe}


class Synthetic(ReadableMultiLevelDatasetBase):
    """
    #TODO
    """

    def _read_dataset(self):
        return self._read_train_test_dataset(self.dataset_directory, ".png")

    def get_v2s(self, seed=42):
        random.seed(seed)
        return {
            "train": self._v2s_subset(self.dataset["train"]),
            "test": self._v2s_subset(self.dataset["test"])
        }

    def _v2s_subset(self, subset):
        gallery = []
        probe = []

        regex = "{0}_{1}_{2}_{3}_{4}.png".format(r"\d{5}",  # Model
                                                 r"A",      # Lighting
                                                 r"270",    # Azimuth
                                                 r"90",     # Zenith
                                                 r"256")    # Resolution
        hq_regex = re.compile(regex)

        for label, paths in subset.as_target_to_source_list().items():
            gallery_candidate = [i for i in paths
                                 if hq_regex.match(os.path.basename(i))][0]
            paths.remove(gallery_candidate)
            gallery.append((label, gallery_candidate))
            probe.extend((label, p) for p in paths)
        return {"gallery": gallery, "probe": probe}
