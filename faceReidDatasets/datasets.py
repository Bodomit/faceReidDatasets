import os
import abc
import itertools
import sacred

from typing import List, Set, Tuple

ingrediant = sacred.Ingredient("dataset")


class DatasetBase(abc.ABC):
    """
    #TODO
    """

    def __init__(self, dataset: List[Tuple[str, str]]):
        self._dataset = sorted(dataset, key=lambda x: (x[1], x[0]))

    @property
    def dataset(self):
        return self._dataset

    def __iter__(self):
        return self.dataset.__iter__()

    def as_target_to_source_list(self):
        target_to_source_list = {}
        for k, g in itertools.groupby(self._dataset, lambda x: x[1]):
            target_to_source_list[k] = [i[0] for i in g]
        return target_to_source_list


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


class VGGFace2(MutiLevelDatasetBase):
    """
    #TODO
    """

    def __init__(self, dataset_directory):
        dataset_directory = os.path.expanduser(dataset_directory)
        dataset_directory = os.path.abspath(dataset_directory)
        self.dataset_directory = dataset_directory
        super().__init__(self._read_dataset())

    def _read_dataset(self):
        if not os.path.isdir(self.dataset_directory):
            raise NotADirectoryError()

        dataset = {
            "train": [],
            "test": []
        }

        for root, dirs, files in os.walk(self.dataset_directory, topdown=True):
            assert not (dirs and files)

            for file in files:
                if os.path.splitext(file)[1] == ".jpg":
                    path = os.path.join(root, file)
                    path = os.path.expanduser(path)
                    path = os.path.abspath(path)
                    label = os.path.basename(os.path.dirname(path))
                    subset = os.path.basename(os.path.dirname(root))
                    dataset[subset].append((path, label))

        return dataset
