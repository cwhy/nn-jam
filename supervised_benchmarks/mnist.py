from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NewType, Literal, Any, TypeVar, Generic, Final, Dict

import numpy as np
from variables import VariableGroup, VariableTensor, OneHot, Bounded

from supervised_benchmarks import dataset_utils
from supervised_benchmarks.download_utils import check_integrity
from supervised_benchmarks.dataset_utils import download_resources, get_raw_path
from supervised_benchmarks.protocols import Data, DataEnv, DataConfig, Dataset

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
mnist_in = VariableGroup(name="mnist_in",
                         variables={
                             (Bounded(max=1, min=0), (28, 28))
                         })
mnist_out = VariableGroup(name="mnist_out",
                          variables={
                              (OneHot(n_category=10), (1,))
                          })

name: Literal["MNIST"] = "MNIST"
Flat = Literal["Flat"]
Train = Literal["Train"]
Test = Literal["Test"]


class MnistDataConfig(NamedTuple):
    base_path: Path
    supervised: bool = True
    shuffle: bool = False
    type: Literal['DataConfig'] = 'DataConfig'


def download_mnist_(base_path: Path) -> None:
    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    download_resources(base_path, name, resources, mirrors)


FeatureTypeTag = TypeVar('FeatureTypeTag')
SubsetTypeTag = TypeVar('SubsetTypeTag')


@dataclass(frozen=True)
class MnistData(Generic[FeatureTypeTag, SubsetTypeTag]):
    content: np.ndarray
    feature_type: FeatureTypeTag
    subset_type: SubsetTypeTag


class FlatMnistEnv(Generic[SubsetTypeTag]):
    def __init__(self,
                 subset: SubsetTypeTag,
                 data_train: np.ndarray,
                 data_test: np.ndarray) -> None:
        self.data_train = data_train
        ...

    @property
    def input(self) -> MnistData[Flat, SubsetTypeTag]:
        if subset == Literal["Train"]
        return self._input

    @property
    def output(self) -> MnistData[Flat, SubsetTypeTag]:
        return self._output

    @property
    def input_protocol(self) -> VariableGroup:
        return mnist_in

    @property
    def output_protocol(self) -> VariableGroup:
        return mnist_out

def preprocess_mnist_(data_config: MnistDataConfig) -> Dict[str, np.ndarray]:
    """

    :param data_config:
    :return:
    Cached files
    """
    ...

class Mnist:
    def __init__(self, data_config: MnistDataConfig) -> None:
        download_mnist_(data_config.base_path)
        self.data = preprocess_mnist_(dataconfig)

    @property
    def name(self) -> Literal['MNIST']:
        return name

    @property
    def train(self) -> FlatMnistEnv:
        return FlatMnistEnv(input=MnistData(), output=MnistData())

    @property
    def test(self) -> FlatMnistEnv:
        return FlatMnistEnv(input=MnistData(), output=MnistData())
