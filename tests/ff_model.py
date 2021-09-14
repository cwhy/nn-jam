from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Protocol, Mapping, FrozenSet, NamedTuple, Literal, Generic

import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad
from jax.scipy.special import logsumexp
from numpy.typing import NDArray
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar

from supervised_benchmarks.dataset_protocols import Input, Output, Port, DataContent, \
    DataPool, DataContentCov, DataContentContra
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.metric_protocols import MetricResult, PairMetric, ResultContentCov
from supervised_benchmarks.metrics import get_mean_acc
from supervised_benchmarks.mnist import MnistDataConfig, Mnist, FixedTrain, FixedTest
from supervised_benchmarks.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.protocols import ModelConfig
from supervised_benchmarks.sampler import Sampler, FullBatchSampler, FullBatchSamplerConfig, \
    FixedEpochSamplerConfig


class Prepare(Protocol[DataContent]):
    def __call__(self, model_config: ModelConfig, train_sampler: Sampler[DataContent]) -> Model[DataContent]: ...


class Benchmark(Protocol[DataContent]):
    @property
    @abstractmethod
    def sampler(self) -> Sampler[DataContent]: ...

    @property
    @abstractmethod
    def metrics(self) -> Mapping[Port, PairMetric[DataContent]]: ...


@dataclass(frozen=True)
class BenchmarkImp(Generic[DataContent]):
    sampler: Sampler[DataContent]
    _metrics: Mapping[Port, PairMetric[DataContent]]

    @property
    def metrics(self) -> Mapping[Port, PairMetric[DataContent]]:
        return self._metrics


class Model(Protocol[DataContent]):

    @property
    @abstractmethod
    def repertoire(self) -> FrozenSet[Port]:
        """
        Output ports
        """
        ...

    @property
    @abstractmethod
    def ports(self) -> Mapping[Port, Variable]:
        """
        Variable Protocol of different ports
        """
        ...

    def perform(self, data_src: Mapping[Port, DataContent], tgt: Port) -> DataContent: ...

    def perform_batch(self,
                      data_src: Mapping[Port, DataContent],
                      tgt: FrozenSet[Port]) -> Mapping[Port, DataContent]: ...


def measure_model(model: Model, benchmark: Benchmark) -> List[MetricResult]:
    sampler: Sampler = benchmark.sampler
    assert all((k in model.repertoire) for k in benchmark.metrics)

    if sampler.tag == 'FullBatchSampler':
        assert isinstance(sampler, FullBatchSampler)
        return [
            metric.measure(
                model.perform(sampler.full_batch, tgt),
                sampler.full_batch[tgt])
            for tgt, metric in benchmark.metrics.items()]
    else:
        raise NotImplementedError


def measure(measurement_fn, predict_fn, port_pair: Tuple[Port, Port], sampler: Sampler) -> float:
    src, tgt = port_pair
    if sampler.tag == 'FullBatchSampler':
        assert isinstance(sampler, FullBatchSampler)
        return measurement_fn(predict_fn(sampler.full_batch[src]), sampler.full_batch[tgt])
    else:
        raise NotImplementedError


class MlpModelUtils:
    @staticmethod
    def forward(params, inputs):
        activations = inputs
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jnp.tanh(outputs)

        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return logits - logsumexp(logits, axis=1, keepdims=True)

    @staticmethod
    def _loss(params, batch):
        inputs, targets = batch
        preds = MlpModelUtils.forward(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    @staticmethod
    @jit
    def update(params, step_size, batch):
        grads = grad(MlpModelUtils._loss)(params, batch)
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, grads)]

    # noinspection PyTypeChecker
    # because pycharm sucks
    @staticmethod
    def prepare(config: MlpModelConfig,
                pool_dict: Mapping[Port, DataPool[NDArray]]) -> Model[NDArray]:
        rng = npr.RandomState(0)
        param_scale = 0.01
        params = [(param_scale * rng.standard_normal((m, n)), param_scale * rng.standard_normal(n))
                  for m, n, in zip(config.layer_sizes[:-1], config.layer_sizes[1:])]
        model = MlpModel(config, params)
        train_pool = subset_all(pool_dict, FixedTrain)
        test_pool = subset_all(pool_dict, FixedTest)
        train_sampler = FixedEpochSamplerConfig(config.train_batch_size).get_sampler(train_pool)
        benchmark_tr = BenchmarkImp(
            sampler=FullBatchSamplerConfig().get_sampler(train_pool),
            _metrics={
                Output: get_mean_acc(10)
            }
        )
        benchmark_tst = BenchmarkImp(
            sampler=FullBatchSamplerConfig().get_sampler(test_pool),
            _metrics={
                Output: get_mean_acc(10)
            }
        )
        for epoch in range(config.num_epochs):
            start_time = time.time()
            for _ in range(train_sampler.num_batches):
                model.update_(next(train_sampler.iter[Input]), next(train_sampler.iter[Output]))
            epoch_time = time.time() - start_time

            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            trrs = measure_model(model, benchmark_tr)
            tstrs = measure_model(model, benchmark_tst)
            print(f"Training result {trrs}")
            print(f"Test set result {tstrs}")
        return model


class MlpModelConfig(NamedTuple):
    step_size: float
    num_epochs: int
    train_batch_size: int
    layer_sizes: List[int]
    type: Literal['ModelConfig'] = 'ModelConfig'


@dataclass
class MlpModel:
    config: MlpModelConfig
    params: List[Tuple[NDArray, NDArray]]

    repertoire: FrozenSet[Port] = frozenset([Output])
    # noinspection PyTypeChecker
    # because pycharm sucks
    ports: Mapping[Port, Variable] = field(default_factory=lambda: {
        Output: var_scalar(one_hot(10)),
        Input: var_tensor(
            gaussian(0, 1),
            dims={dim('features', 28 * 28, positioned=True)})
    })

    def update_(self, x: NDArray, y: NDArray):
        self.params = MlpModelUtils.update(self.params, self.config.step_size, (x, y))

    def predict(self, inputs: NDArray):
        return MlpModelUtils.forward(self.params, inputs)

    def perform(self, data_src: Mapping[Port, NDArray], tgt: Port) -> NDArray:
        assert Input in data_src and tgt == Output
        return self.predict(data_src[Input])

    def perform_batch(self,
                      data_src: Mapping[Port, NDArray],
                      tgt: FrozenSet[Port]) -> Mapping[Port, NDArray]:
        assert Input in data_src and len(tgt) == 1 and next(iter(tgt)) == Output
        return {Output: self.predict(data_src[Input])}


config_ = MlpModelConfig(
    step_size=0.03,
    num_epochs=20,
    train_batch_size=32,
    layer_sizes=[784, 784, 256, 10]
)


def accuracy(output: NDArray, target: NDArray) -> float:
    target_class = jnp.argmax(target, axis=1)
    output_class = jnp.argmax(output, axis=1)
    return jnp.mean(output_class == target_class)


k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))
mnist_in_flattened = MnistConfigIn(is_float=True, is_flat=True).get_var()
mnist_out_1hot = MnistConfigOut(is_1hot=True).get_var()
pool_dict_ = k.retrieve({Input: mnist_in_flattened, Output: mnist_out_1hot})
# noinspection PyTypeChecker
# because pycharm sucks
model_ = MlpModelUtils.prepare(config_, pool_dict_)

# noinspection PyTypeChecker
# because pycharm sucks
test_pool = subset_all(pool_dict_, FixedTest)
test_sampler = FullBatchSamplerConfig().get_sampler(test_pool)
benchmark = BenchmarkImp(
    sampler=test_sampler,
    _metrics={
        Output: get_mean_acc(10)
    }
)
# noinspection PyTypeChecker
# because pycharm sucks
z = measure_model(model_, benchmark)
print(z)
