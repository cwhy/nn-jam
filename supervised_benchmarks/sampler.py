from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, Iterator, Mapping, Protocol, runtime_checkable, TypeVar, NamedTuple

import numpy.random as npr

from supervised_benchmarks.dataset_protocols import DataContentCov, Port, Data
from supervised_benchmarks.mnist import FixedTrain

SamplerType = Literal['FixedEpochSampler', 'FullBatchSampler', 'MiniBatchSampler']


class Sampler(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def tag(self) -> SamplerType: ...


class SamplerConfig(Protocol):
    @property
    @abstractmethod
    def sampler_tag(self) -> SamplerType: ...


@runtime_checkable
class MiniBatchSampler(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def iter(self) -> Mapping[Port, Iterator[DataContentCov]]: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler', 'MiniBatchSampler']: ...


@runtime_checkable
class FullBatchSampler(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def full_batch(self) -> Mapping[Port, DataContentCov]: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FullBatchSampler']: ...


@runtime_checkable
class FixedEpochSampler(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def iter(self) -> Mapping[Port, Iterator[DataContentCov]]: ...

    @property
    @abstractmethod
    def num_batches(self) -> int: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler']: ...


@runtime_checkable
class FixedEpochSamplerConfig(Protocol):
    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abstractmethod
    def sampler_tag(self) -> Literal['FixedEpochSampler']: ...


@dataclass(frozen=True)
class FixedEpochSamplerImp(Generic[DataContentCov]):
    num_batches: int
    _iter: Mapping[Port, Iterator[DataContentCov]]

    @property
    def iter(self) -> Mapping[Port, Iterator[DataContentCov]]:
        return self._iter

    @property
    def tag(self) -> Literal['FixedEpochSampler']:
        return 'FixedEpochSampler'


@runtime_checkable
class FullBatchSamplerConfig(Protocol):
    @property
    @abstractmethod
    def sampler_tag(self) -> Literal['FullBatchSampler']: ...


@dataclass(frozen=True)
class FullBatchSamplerImp(Generic[DataContentCov]):
    _batch: Mapping[Port, DataContentCov]

    @property
    def full_batch(self) -> Mapping[Port, DataContentCov]:
        return self._batch

    @property
    def tag(self) -> Literal['FullBatchSampler']:
        return 'FullBatchSampler'


def get_full_batch_sampler(data_dict: Mapping[Port, Data[DataContentCov]]) -> FullBatchSampler[DataContentCov]:
    return FullBatchSamplerImp({k: v.content for k, v in data_dict.items()})


def get_fixed_epoch_sampler(batch_size: int,
                            data_dict: Mapping[Port, Data[DataContentCov]]) -> FixedEpochSampler[DataContentCov]:
    num_train = len(FixedTrain.indices)
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + int(bool(leftover))

    def data_stream(data: Data[DataContentCov]) -> Iterator[DataContentCov]:
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
