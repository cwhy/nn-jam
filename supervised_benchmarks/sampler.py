# Samplers are used to sample a subset of the data.
# Takes a DataSubset and returns an iterator over DataUnits

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Iterator, Protocol, runtime_checkable, TypeVar, NamedTuple

import numpy.random as npr

from supervised_benchmarks.dataset_protocols import DataSubset, DataUnit

SamplerType = Literal['FixedEpochSampler', 'FullBatchSampler', 'MiniBatchSampler']
MiniBatchSamplerType = Literal['FixedEpochSampler', 'MiniBatchSampler']
SamplerTypeVar = TypeVar('SamplerTypeVar', bound=SamplerType, covariant=True)


class Sampler(Protocol):
    @property
    @abstractmethod
    def tag(self) -> SamplerType: ...


@runtime_checkable
class MiniBatchSampler(Protocol):
    @property
    @abstractmethod
    def iter(self) -> Iterator[DataUnit]: ...

    @property
    @abstractmethod
    def tag(self) -> MiniBatchSamplerType: ...


@runtime_checkable
class FullBatchSampler(Protocol):
    @property
    @abstractmethod
    def full_batch(self) -> DataUnit: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FullBatchSampler']: ...


@runtime_checkable
class FixedEpochSampler(Protocol):
    @property
    @abstractmethod
    def iter(self) -> Iterator[DataUnit]: ...

    @property
    @abstractmethod
    def num_batches(self) -> int: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler']: ...


@dataclass(frozen=True)
class FixedEpochSamplerImp:
    num_batches: int
    _iter: Iterator[DataUnit]

    @property
    def iter(self) -> Iterator[DataUnit]:
        return self._iter

    @property
    def tag(self) -> Literal['FixedEpochSampler']:
        return 'FixedEpochSampler'


@dataclass(frozen=True)
class FullBatchSamplerImp:
    _batch: DataUnit

    @property
    def full_batch(self) -> DataUnit:
        return self._batch

    @property
    def tag(self) -> Literal['FullBatchSampler']:
        return 'FullBatchSampler'


class SamplerConfig(Protocol[SamplerTypeVar]):
    @property
    @abstractmethod
    def sampler_tag(self) -> SamplerTypeVar: ...

    def get_sampler(self,
                    data_subset: DataSubset
                    ) -> Sampler:
        ...


class FixedEpochSamplerConfig(NamedTuple):
    batch_size: int
    sampler_tag: Literal['FixedEpochSampler'] = 'FixedEpochSampler'

    def get_sampler(self,
                    data_subset: DataSubset) -> FixedEpochSampler:
        batch_size = self.batch_size
        num_data = data_subset.subset.len
        num_complete_batches, leftover = divmod(num_data, batch_size)
        num_batches = num_complete_batches + int(bool(leftover))

        def data_stream() -> Iterator[DataUnit]:
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_data)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield {p: arr[batch_idx] for p, arr in data_subset.content_map.items()}

        return FixedEpochSamplerImp(num_batches, data_stream())


class FullBatchSamplerConfig(NamedTuple):
    sampler_tag: Literal['FullBatchSampler'] = 'FullBatchSampler'

    @staticmethod
    def get_sampler(data_subset: DataSubset) -> FullBatchSampler:
        return FullBatchSamplerImp(data_subset.content_map)
