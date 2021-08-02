from __future__ import annotations
from pathlib import Path
from typing import NamedTuple, Literal, TypeVar, Dict

import numpy as np

from supervised_benchmarks.dataset_protocols import Port, Subset, DataQuery
from supervised_benchmarks.dataset_utils import download_resources, get_data_dir
from supervised_benchmarks.mnist_utils import read_sn3_pascalvincent_ndarray

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']


name: Literal["MNIST"] = "MNIST"
Flat = Literal["Flat"]


class MnistDataConfig(NamedTuple):
    base_path: Path
    type: Literal['DataConfig'] = 'DataConfig'


def get_mnist_(base_path: Path) -> Dict[str, np.ndarray]:
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

    data = {
        ".".join(f_name.split("-")[:2]):
            read_sn3_pascalvincent_ndarray(
                get_data_dir(base_path, name, 'raw').joinpath(f_name.split(".")[0])
            )
        for f_name, _ in resources
    }
    np.savez(get_data_dir(base_path, name, 'processed').joinpath('array_dict'), data)
    return data


class MnistData(NamedTuple):
    port: Port
    protocol: VariablePort
    subset: Subset
    content: np.ndarray


class MnistDataPool(NamedTuple):
    array_dict: Dict[str, np.ndarray]
    port: Port
    target_protocol: VariablePort

    def subset(self, subset: Subset) -> MnistData:
        return MnistData(self.port, self.target_protocol, subset, np.zeros(0))


class Mnist:
    def __init__(self, data_config: MnistDataConfig) -> None:
        self.array_dict: Dict[str, np.ndarray] = get_mnist_(data_config.base_path)

    @property
    def name(self) -> Literal['MNIST']:
        return name

    def retrieve(self, query: DataQuery) -> Dict[Port, MnistDataPool]:
        return {
            port: MnistDataPool(self.array_dict, port, variable_protocol)
            for port, variable_protocol in query.items()
        }
