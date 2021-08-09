from __future__ import annotations
from pathlib import Path
from typing import NamedTuple, Literal, TypeVar, Dict, Tuple, Callable

import numpy as np
from einops import rearrange

from variable_protocols.variables import Variable
from supervised_benchmarks.dataset_protocols import Port, Subset, DataQuery, Input, Output, FixedTrain, FixedTest
from supervised_benchmarks.dataset_utils import download_resources, get_data_dir
from supervised_benchmarks.mnist_utils import read_sn3_pascalvincent_ndarray
from variable_protocols.variables import dim, bounded_float, var_tensor, var_scalar, one_hot

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

name: Literal["MNIST"] = "MNIST"
# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in: Variable = var_tensor(bounded_float(0, 1), {dim("h", 28), dim("w", 28)})
# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in_flattened: Variable = var_tensor(bounded_float(0, 1), {dim("hw", 28 * 28)})
# noinspection PyTypeChecker
# because pyCharm sucks
mnist_out = var_scalar(one_hot(10))

transformations: Dict[Tuple[Variable, Variable], Callable[[np.ndarray], np.ndarray]] = {
    (mnist_in, mnist_in_flattened): lambda x: rearrange(x, 'b h w -> b (h w)'),
    (mnist_in_flattened, mnist_in): lambda x: rearrange(x, 'b (h w) -> b h w', h=28, w=28)
}


def get_transformations(protocols: Tuple[Variable, Variable]) -> Callable[[np.ndarray], np.ndarray]:
    s, t = protocols
    # TODO after support struct-check
    if s == t:
        return lambda x: x
    else:
        return transformations[(s, t)]


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
    protocol: Variable
    subset: Subset
    content: np.ndarray


class MnistDataPool(NamedTuple):
    array_dict: Dict[str, np.ndarray]
    port: Port
    src_var: Variable
    tgt_var: Variable

    def subset(self, subset: Subset) -> MnistData:
        if self.port == Input and subset == FixedTrain:
            tag = 'train.images'
        elif self.port == Input and subset == FixedTest:
            tag = 'test.images'
        elif self.port == Output and subset == FixedTrain:
            tag = 'train.labels'
        else:
            assert self.port == Output and subset == FixedTest
            tag = 'test.labels'
        target = get_transformations((self.src_var, self.tgt_var))(self.array_dict[tag])
        return MnistData(self.port, self.tgt_var, subset, target)


class Mnist:
    def __init__(self, data_config: MnistDataConfig) -> None:
        self.array_dict: Dict[str, np.ndarray] = get_mnist_(data_config.base_path)
        self.protocols: Dict[str, Variable] = {
            Input: mnist_in,
            Output: mnist_out
        }

    @property
    def name(self) -> Literal['MNIST']:
        return name

    def retrieve(self, query: DataQuery) -> Dict[Port, MnistDataPool]:
        return {
            port: MnistDataPool(
                self.array_dict,
                port,
                src_var=self.protocols[port],
                tgt_var=variable_protocol)
            for port, variable_protocol in query.items()
        }
