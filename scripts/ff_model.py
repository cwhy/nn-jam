# test mnist in audition, without references to jax-make
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Mapping, FrozenSet, Literal

import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad
from jax.scipy.special import logsumexp
from numpy.typing import NDArray
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import PortSpecs, DataConfig
from supervised_benchmarks.ports import Port, Input, Output
from supervised_benchmarks.metrics import get_mean_acc, get_pair_metric
from supervised_benchmarks.mnist.mnist import MnistDataConfig, FixedTrain, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.model_utils import Train
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import MiniBatchSampler


@dataclass(frozen=True)
class MlpModelConfig:
    step_size: float
    num_epochs: int
    train_batch_size: int
    layer_sizes: List[int]
    train_data_config: DataConfig
    repertoire: FrozenSet[Port] = frozenset([Output])
    # noinspection PyTypeChecker
    # because pycharm sucks
    ports: Mapping[Port, Variable] = field(default_factory=lambda: {
        Output: var_scalar(one_hot(10)),
        Input: var_tensor(
            gaussian(0, 1),
            dims={dim('features', 28 * 28, positioned=True)})
    })
    type: Literal['ModelConfig'] = 'ModelConfig'

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
        preds = MlpModelConfig.forward(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    @staticmethod
    @jit
    def update(params, step_size, batch):
        grads = grad(MlpModelConfig._loss)(params, batch)
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, grads)]

    # noinspection PyTypeChecker
    # because pycharm sucks
    def prepare(self) -> Performer[NDArray]:
        rng = npr.RandomState(0)
        param_scale = 0.01
        params = [(param_scale * rng.standard_normal((m, n)), param_scale * rng.standard_normal(n))
                  for m, n, in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        model = MlpModel(self, params)
        Train(
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            bench_configs=[BenchmarkConfig(metrics={Output: get_pair_metric('mean_acc', var_scalar(one_hot(10)))}, on=FixedTrain),
                           BenchmarkConfig(metrics={Output: get_pair_metric('mean_acc', var_scalar(one_hot(10)))}, on=FixedTest)],
            model=model,
            data_subset=FixedTrain,
            data_config=self.train_data_config
        ).run_()
        return model


@dataclass
class MlpModel:
    model: MlpModelConfig
    params: List[Tuple[NDArray, NDArray]]

    def update_(self, sampler: MiniBatchSampler):
        self.params = MlpModelConfig.update(self.params,
                                            self.model.step_size,
                                            (next(sampler.iter[Input]), next(sampler.iter[Output])))

    def predict(self, inputs: NDArray):
        return MlpModelConfig.forward(self.params, inputs)

    def perform(self, data_src: Mapping[Port, NDArray], tgt: Port) -> NDArray:
        assert Input in data_src and tgt == Output
        return self.predict(data_src[Input])

    def perform_batch(self,
                      data_src: Mapping[Port, NDArray],
                      tgt: FrozenSet[Port]) -> Mapping[Port, NDArray]:
        assert Input in data_src and len(tgt) == 1 and next(iter(tgt)) == Output
        return {Output: self.predict(data_src[Input])}


data_config_ = MnistDataConfig(
    base_path=Path('/Data/torchvision/'),
    port_vars={
        Input: MnistConfigIn(is_float=True, is_flat=True).get_var(),
        Output: MnistConfigOut(is_1hot=True).get_var()
    })

benchmark_config_ = BenchmarkConfig(
    metrics={Output: get_pair_metric('mean_acc', var_scalar(one_hot(10)))},
    on=FixedTest)

# noinspection PyTypeChecker
# Because Pycharm sucks
model_ = MlpModelConfig(
    step_size=0.03,
    num_epochs=20,
    train_batch_size=32,
    layer_sizes=[784, 784, 256, 10],
    train_data_config=data_config_
).prepare()
# noinspection PyTypeChecker
# Because Pycharm sucks
z = benchmark_config_.bench(data_config_, model_)
print(z)
