from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict

import numpy as np
from numpy.typing import NDArray
from variable_protocols.variables import Variable

from supervised_benchmarks.dataset_protocols import Subset, PortSpecs, FixedSubset, DataSubset, FixedSubsetType, \
    FixedTrain, FixedTest
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


def get_mnist_(base_path: Path) -> Dict[str, NDArray]:
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

    data: dict[str, NDArray] = {
        ".".join(f_name.split("-")[:2]):
            read_sn3_pascalvincent_ndarray(
                get_data_dir(base_path, name, 'raw').joinpath(f_name.split(".")[0])
            )
        for f_name, _ in resources
    }
    # TODO: check if there
    # np.savez(get_data_dir(base_path, name, 'processed').joinpath('array_dict'), **data)
    return data


class MnistDataPool(NamedTuple):
    array_dict: Mapping[str, NDArray]
    fixed_subsets: Mapping[FixedSubsetType, DataSubset]
    query: PortSpecs
    raw_specs: PortSpecs

    def subset(self, subset: Subset) -> DataSubset:
        ...
        # transform = get_transformations((self.src_var, self.tgt_var))
        # port_tag = 'images' if self.port is Input else 'labels'
        # target = transform(self.array_dict[f"all.{port_tag}"][subset.indices])
        # # noinspection PyTypeChecker
        # # Because pycharm sucks
        # return MnistData(self.port, self.tgt_var, subset, target)


mnist_in_raw = MnistConfigIn(is_float=False, is_flat=False).get_var()
mnist_out_raw = MnistConfigOut(is_1hot=False).get_var()


class Mnist:
    @property
    def exports(self) -> FrozenSet[Port]:
        return frozenset({Input, Output})

    def __init__(self, base_path: Path) -> None:
        self.array_dict: Dict[str, NDArray] = get_mnist_(base_path)
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
        self.protocols: Mapping[Port, Variable] = {
            Input: mnist_in_raw,
            Output: mnist_out_raw
        }

    @property
    def name(self) -> Literal['MNIST']:
        return name

    def get_fixed_datasets(self, query: PortSpecs) -> Mapping[FixedSubsetType, DataSubset]:
        assert set(query.keys()).issubset(self.exports)

        def get_data(port: Port, is_train: bool) -> NDArray:
            # TODO test, may not right
            prefix = 'train' if is_train else 't10k'
            if port is Input:
                return self.array_dict[f'{prefix}.images']
            elif port is Output:
                return self.array_dict[f'{prefix}.labels']
            else:
                raise ValueError(f'Unknown port {port}')

        fixed_datasets: Dict[FixedSubsetType, DataSubset] = {
            FixedTrain: DataSubset(query,
                                   FixedSubset(FixedTrain, n_samples_tr),
                                   {port: get_data(port, is_train=True) for port in query}),
            FixedTest: DataSubset(query,
                                  FixedSubset(FixedTest, n_samples_tst),
                                  {port: get_data(port, is_train=False) for port in query})
        }
        return fixed_datasets

    def retrieve(self, query: PortSpecs) -> MnistDataPool:
        assert all(port in self.exports for port in query)

        return MnistDataPool(
            array_dict=self.array_dict,
            fixed_subsets=self.get_fixed_datasets(query),
            query=query)


class MnistDataConfig(NamedTuple):
    base_path: Path
    query: PortSpecs
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self) -> MnistDataPool:
        return Mnist(self.base_path).retrieve(self.query)
