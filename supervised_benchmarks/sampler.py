from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, Iterator, Dict, Protocol, runtime_checkable

import numpy.random as npr

from supervised_benchmarks.dataset_protocols import DataContent, Port, Data
from supervised_benchmarks.mnist import FixedTrain


class Sampler(Protocol):
    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler', 'FullBatchSampler', 'MiniBatchSampler']: ...


@runtime_checkable
class MiniBatchSampler(Protocol[DataContent]):
    @property
    @abstractmethod
    def iter(self) -> Dict[Port, Iterator[DataContent]]: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler', 'MiniBatchSampler']: ...


@runtime_checkable
class FullBatchSampler(Protocol[DataContent]):
    @property
    @abstractmethod
    def full_batch(self) -> Dict[Port, DataContent]: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FullBatchSampler']: ...


@runtime_checkable
class FixedEpochSampler(Protocol[DataContent]):
    @property
    @abstractmethod
    def iter(self) -> Dict[Port, Iterator[DataContent]]: ...

    @property
    @abstractmethod
    def num_batches(self) -> int: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler']: ...


@dataclass(frozen=True)
class FixedEpochSamplerImp(Generic[DataContent]):
    num_batches: int
    _iter: Dict[Port, Iterator[DataContent]]

    @property
    def iter(self) -> Dict[Port, Iterator[DataContent]]:
        return self._iter

    @property
    def tag(self) -> Literal['FixedEpochSampler']:
        return 'FixedEpochSampler'


def get_fixed_epoch_sampler(batch_size: int,
                            data_dict: Dict[Port, Data[DataContent]]) -> FixedEpochSampler[DataContent]:
    num_train = len(FixedTrain.indices)
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + int(bool(leftover))

    def data_stream(data: Data[DataContent]) -> Iterator[DataContent]:
        content = data.content
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield content[batch_idx]

    # noinspection PyTypeChecker
    # because pyCharm sucks
    return FixedEpochSamplerImp(num_batches, {k: data_stream(v) for k, v in data_dict.items()})


@dataclass(frozen=True)
class FullBatchSamplerImp(Generic[DataContent]):
    _batch: Dict[Port, DataContent]

    @property
    def full_batch(self) -> Dict[Port, DataContent]:
        return self._batch

    @property
    def tag(self) -> Literal['FullBatchSampler']:
        return 'FullBatchSampler'


def get_full_batch_sampler(data_dict: Dict[Port, Data[DataContent]]) -> FullBatchSampler[DataContent]:
    return FullBatchSamplerImp({k: v.content for k, v in data_dict.items()})
