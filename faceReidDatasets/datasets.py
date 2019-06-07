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

    def __init__(self, dataset: List[Set[Tuple[str, str]]]):
        self._dataset = sorted(dataset, key=lambda x: x[1])

    @property
    def dataset(self):
        return self._dataset

    def __iter__(self):
        return self.dataset.__iter__

    def as_target_to_source_list(self):
        return dict(itertools.groupby(self._dataset, lambda x: x[1]))