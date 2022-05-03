from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict

import numpy as np
import numpy.typing as npt
from variable_protocols.variables import Variable

from supervised_benchmarks.dataset_protocols import Subset, DataQuery, FixedSubset, DataSubset, DataPortMap
from supervised_benchmarks.ports import Port, Input, Output
from supervised_benchmarks.dataset_utils import download_resources, get_data_dir
from supervised_benchmarks.mnist.mnist_utils import read_sn3_pascalvincent_ndarray
from supervised_benchmarks.mnist.mnist_variations import get_transformations, MnistConfigIn, MnistConfigOut

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

name: Literal["MNIST"] = "MNIST"

n_samples_tr = 60000
n_samples_tst = 10000
n_samples = n_samples_tr + n_samples_tst

FixedTrain = FixedSubset('FixedTrain', list(range(n_samples_tr)))
FixedTest = FixedSubset('FixedTest', list(range(n_samples_tr, n_samples)))
FixedAll = FixedSubset('All', list(range(n_samples)))


def get_mnist_(base_path: Path) -> Dict[str, npt.NDArray]:
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
    # TODO: check if there
    np.savez(get_data_dir(base_path, name, 'processed').joinpath('array_dict'), data)
    return data


class MnistData(NamedTuple):
    port: Port
    protocol: Variable
    subset: Subset
    content: npt.NDArray


class MnistDataPool(NamedTuple):
    array_dict: Mapping[str, npt.NDArray]
    port: Port
    src_var: Variable
    tgt_var: Variable

    def subset(self, subset: Subset) -> DataSubset[npt.NDArray]:
        transform = get_transformations((self.src_var, self.tgt_var))
        port_tag = 'images' if self.port is Input else 'labels'
        target = transform(self.array_dict[f"all.{port_tag}"][subset.indices])
        # noinspection PyTypeChecker
        # Because pycharm sucks
        return MnistData(self.port, self.tgt_var, subset, target)


mnist_in_raw = MnistConfigIn(is_float=False, is_flat=False).get_var()
mnist_out_raw = MnistConfigOut(is_1hot=False).get_var()


class Mnist:
    @property
    def ports(self) -> FrozenSet[Port]:
        return frozenset({Input, Output})

    def __init__(self, base_path: Path) -> None:
        self.array_dict: Dict[str, npt.NDArray] = get_mnist_(base_path)
        assert n_samples_tr == self.array_dict['train.images'].shape[0]
        assert n_samples_tst == self.array_dict['t10k.images'].shape[0]
        assert n_samples_tr == len(self.array_dict['train.labels'])
        assert n_samples_tst == len(self.array_dict['t10k.labels'])
        self.array_dict['all.images'] = np.concatenate((self.array_dict['train.images'],
                                                        self.array_dict['t10k.images']), axis=0)
        self.array_dict['all.labels'] = np.concatenate((self.array_dict['train.labels'],
                                                        self.array_dict['t10k.labels']), axis=0)
        del self.array_dict['train.images']
        del self.array_dict['train.labels']
        del self.array_dict['t10k.images']
        del self.array_dict['t10k.labels']
        self.protocols: Mapping[str, Variable] = {
            Input: mnist_in_raw,
            Output: mnist_out_raw
        }

    @property
    def name(self) -> Literal['MNIST']:
        return name

    def retrieve(self, query: DataQuery) -> Mapping[Port, MnistDataPool]:
        assert all(port in self.ports for port in query)
        return {
            port: MnistDataPool(
                self.array_dict,
                port,
                src_var=self.protocols[port],
                tgt_var=variable_protocol)
            for port, variable_protocol in query.items()
        }


# noinspection PyTypeChecker
# Because pycharm sucks
class MnistDataConfig(NamedTuple):
    base_path: Path
    port_vars: DataQuery
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self) -> DataPortMap:
        return Mnist(self.base_path).retrieve(self.port_vars)
