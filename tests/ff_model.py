from __future__ import annotations
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Protocol, Mapping, Callable, Any, Set, FrozenSet, TypeVar, NamedTuple, Literal

import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad
from jax.scipy.special import logsumexp
from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import Input, Output, Port, Data, DataContentContra, DataContent, \
    DataContentCov
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.metric_protocols import Metric, MetricResult
from supervised_benchmarks.mnist import MnistDataConfig, Mnist, FixedTrain, FixedTest
from supervised_benchmarks.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.protocols import ModelConfig
from supervised_benchmarks.sampler import get_fixed_epoch_sampler, get_full_batch_sampler, Sampler, FullBatchSampler


class Measure(Protocol[DataContentContra]):
    def __call__(self, output: DataContentContra, target: DataContentContra) -> MetricResult: ...


class Foster(Protocol[DataContent]):
    def __call__(self, model_config: ModelConfig, train_sampler: Sampler[DataContent]) -> Model[DataContent]: ...


class Benchmark(Protocol[DataContent]):
    @property
    @abstractmethod
    def sampler(self) -> Sampler[DataContent]: ...

    @property
    @abstractmethod
    def repertoire(self) -> FrozenSet[Port]: ...

    @property
    @abstractmethod
    def measurements(self) -> Mapping[Port, Measure[DataContent]]: ...


class Model(Protocol[DataContent]):

    # List when multiple outputs comes out
    @property
    @abstractmethod
    def repertoire(self) -> FrozenSet[Port]: ...

    def perform(self, data_src: Mapping[Port, DataContent], tgt: Port) -> DataContent: ...

    def perform_batch(self,
                      data_src: Mapping[Port, DataContent],
                      tgt: FrozenSet[Port]) -> Mapping[Port, DataContent]: ...


def measure_model(model: Model, benchmark: Benchmark) -> List[MetricResult]:
    sampler: Sampler = benchmark.sampler
    assert all((k in model.repertoire) for k in benchmark.repertoire)

    if sampler.tag == 'FullBatchSampler':
        assert isinstance(sampler, FullBatchSampler)
        return [fn(model.perform(sampler.full_batch, tgt),
                   sampler.full_batch[tgt])
                for (srcs, tgt), fn in benchmark.measurements.items()]
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
    def foster(config: MlpModelConfig) -> Model[NDArray]:
        rng = npr.RandomState(0)
        param_scale = 0.01
        params = [(param_scale * rng.standard_normal((m, n)), param_scale * rng.standard_normal(n))
                  for m, n, in zip(config.layer_sizes[:-1], config.layer_sizes[1:])]
        model = MlpModel(config, params)
        k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))
        mnist_in_flattened = MnistConfigIn(is_float=True, is_flat=True).get_var()
        mnist_out_1hot = MnistConfigOut(is_1hot=True).get_var()
        pool_dict = k.retrieve({Input: mnist_in_flattened, Output: mnist_out_1hot})
        train_pool = subset_all(pool_dict, FixedTrain)
        test_pool = subset_all(pool_dict, FixedTest)
        train_sampler = get_fixed_epoch_sampler(config.train_batch_size, train_pool)
        train_all = get_full_batch_sampler(train_pool)
        test_all = get_full_batch_sampler(test_pool)
        flow = (Input, Output)
        for epoch in range(config.num_epochs):
            start_time = time.time()
            for _ in range(train_sampler.num_batches):
                model.update_(next(train_sampler.iter[Input]), next(train_sampler.iter[Output]))
            epoch_time = time.time() - start_time

            train_acc = measure(accuracy, model.predict, flow, train_all)
            test_acc = measure(accuracy, model.predict, flow, test_all)
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))
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
    repertoire: FrozenSet[Port] = frozenset({Output})

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
    step_size=0.01,
    num_epochs=50,
    train_batch_size=32,
    layer_sizes=[784, 784, 256, 10]
)


def accuracy(output: NDArray, target: NDArray) -> float:
    target_class = jnp.argmax(target, axis=1)
    output_class = jnp.argmax(output, axis=1)
    return jnp.mean(output_class == target_class)


model_ = MlpModelUtils.foster(config_)
