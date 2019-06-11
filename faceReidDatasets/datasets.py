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
