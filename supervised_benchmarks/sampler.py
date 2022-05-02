from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, Iterator, Mapping, Protocol, runtime_checkable, TypeVar, NamedTuple

import numpy.random as npr

from supervised_benchmarks.dataset_protocols import DataContentCov, Data
from supervised_benchmarks.ports import Port
from supervised_benchmarks.mnist.mnist import FixedTrain

SamplerType = Literal['FixedEpochSampler', 'FullBatchSampler', 'MiniBatchSampler']
MiniBatchSamplerType = Literal['FixedEpochSampler', 'MiniBatchSampler']
SamplerTypeVar = TypeVar('SamplerTypeVar', bound=SamplerType, covariant=True)


class Sampler(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def tag(self) -> SamplerType: ...


@runtime_checkable
class MiniBatchSampler(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def iter(self) -> Mapping[Port, Iterator[DataContentCov]]: ...

    @property
    @abstractmethod
    def tag(self) -> MiniBatchSamplerType: ...


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


@dataclass(frozen=True)
class FullBatchSamplerImp(Generic[DataContentCov]):
    _batch: Mapping[Port, DataContentCov]

    @property
    def full_batch(self) -> Mapping[Port, DataContentCov]:
        return self._batch

    @property
    def tag(self) -> Literal['FullBatchSampler']:
        return 'FullBatchSampler'


class SamplerConfig(Protocol[SamplerTypeVar]):
    @property
    @abstractmethod
    def sampler_tag(self) -> SamplerTypeVar: ...

    def get_sampler(self,
                    data_dict: Mapping[Port, Data[DataContentCov]]
                    ) -> Sampler[DataContentCov]:
        ...


class FixedEpochSamplerConfig(NamedTuple):
    batch_size: int
    sampler_tag: Literal['FixedEpochSampler'] = 'FixedEpochSampler'

    def get_sampler(self,
                    data_dict: Mapping[Port, Data[DataContentCov]]) -> FixedEpochSampler[DataContentCov]:
        batch_size = self.batch_size
        # TODO refactor Data to contain Map[Port, Content]
        num_data = next(iter(data_dict.values())).subset.len
        num_complete_batches, leftover = divmod(num_data, batch_size)
        num_batches = num_complete_batches + int(bool(leftover))

        def data_stream(data: Data[DataContentCov]) -> Iterator[DataContentCov]:
            content = data.content
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_data)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield content[batch_idx]

        # noinspection PyTypeChecker
        # because pyCharm sucks
        return FixedEpochSamplerImp(num_batches, {k: data_stream(v) for k, v in data_dict.items()})


class FullBatchSamplerConfig(NamedTuple):
    sampler_tag: Literal['FullBatchSampler'] = 'FullBatchSampler'

    @staticmethod
    def get_sampler(data_dict: Mapping[Port, Data[DataContentCov]]) -> FullBatchSampler[DataContentCov]:
        return FullBatchSamplerImp({k: v.content for k, v in data_dict.items()})
