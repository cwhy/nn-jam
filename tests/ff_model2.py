from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce, partial
from operator import mul, add
from pathlib import Path
from typing import Tuple, List, Mapping, FrozenSet, Literal

import jax.nn as jnn
import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad
from jax.scipy.special import logsumexp
from numpy.random import RandomState
from numpy.typing import NDArray
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Input, Output, Port, DataQuery, DataConfig
from supervised_benchmarks.metrics import get_mean_acc
from supervised_benchmarks.mnist import MnistDataConfig, FixedTrain, FixedTest
from supervised_benchmarks.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.model_utils import Train
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import MiniBatchSampler

n_dups = 10
op = add
dropout_keep_rate = 0.8


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
    def forward(params_dups, inputs, train=False, rng=None):
        # params_sum = [(w.sum(axis=0), b.sum(axis=0)) for w, b in params_dups]
        params_sum = [[reduce(op, param)
                       for param in zip(*params)] for params in zip(*params_dups)]
        # print([(a.shape, z.shape) for a, z in params_sum])
        return MlpModelConfig.forward1(params_sum, inputs, train, rng)

    @staticmethod
    def forward1(params, inputs, train, rng: RandomState):
        if train:
            keep = rng.binomial(1, dropout_keep_rate, inputs.shape)
            activations = jnp.where(keep, inputs / dropout_keep_rate, 0)
        else:
            activations = inputs
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jnn.relu(outputs)

        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return logits - logsumexp(logits, axis=1, keepdims=True)

    @staticmethod
    def _loss(params_dups, batch, rng):
        inputs, targets = batch
        preds = MlpModelConfig.forward(params_dups, inputs, train=True, rng=rng)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    @staticmethod
    @partial(jit, static_argnums=4)
    def update_partial(curr_params, step_size, batch, params, i, rng):
        grads = grad(MlpModelConfig._loss)(curr_params, batch, rng)

        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, grads[i])]

    @staticmethod
    @partial(jit, static_argnums=2)
    def get_grads(curr_params, batch, rng):
        return grad(MlpModelConfig._loss)(curr_params, batch, rng)

    @staticmethod
    @jit
    def update_only(params, grads, step_size):
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, grads)]

    @staticmethod
    def update_alt(params_dups, step_size, batch, rng):
        for i in range(n_dups):
            x, y = batch
            ind = y[:, i] == 1
            if sum(ind) > 0:
                new_batch = x[ind, :], y[ind, :]
                grads = MlpModelConfig.get_grads(params_dups, new_batch, rng)
                params_dups[i] = MlpModelConfig.update_only(params_dups[i], grads[i], step_size)
                # params_dups[i] = MlpModelConfig.update_partial(params_dups, step_size, new_batch, params_dups[i], i,
                #                                                rng)
        return params_dups

    @staticmethod
    @partial(jit, static_argnums=3)
    def update(params_dups, step_size, batch, rng):
        grads = grad(MlpModelConfig._loss)(params_dups, batch, rng)
        return [[(w - step_size * dw, b - step_size * db)
                 for (w, b), (dw, db) in zip(params_dups[i], grads[i])]
                for i in range(n_dups)]

    # noinspection PyTypeChecker
    # because pycharm sucks
    def prepare(self) -> Performer[NDArray]:
        rng = npr.RandomState(0)
        if op is mul:
            param_scale = pow(0.5, 1 / n_dups)
        elif op is add:
            param_scale = 0.01 / n_dups
        else:
            param_scale = 0.01

        params_dups = [[(param_scale * rng.standard_normal((m, n)), param_scale * rng.standard_normal(n))
                        for m, n, in zip(self.layer_sizes[:-1], self.layer_sizes[1:])] for _ in range(n_dups)]
        # jax does not support this?
        # params_sum = [[sum(param) for param in zip(*params)] for params in zip(*params_dups)]
        # print([(a.shape, z.shape) for a, z in params_sum])
        # params_sum = [(w.sum(axis=0), b.sum(axis=0)) for w, b in params_dups]
        # print([(a.shape, z.shape) for a, z in params_sum])
        model = MlpModel(self, params_dups, rng)
        Train(
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            bench_configs=[BenchmarkConfig(metrics={Output: get_mean_acc(10)}, on=FixedTrain),
                           BenchmarkConfig(metrics={Output: get_mean_acc(10)})],
            model=model,
            data_subset=FixedTrain,
            data_config=self.train_data_config
        ).run_()
        return model


@dataclass
class MlpModel:
    model: MlpModelConfig
    params_dups: List[List[Tuple[NDArray, NDArray]]]
    rng: RandomState

    def update_(self, sampler: MiniBatchSampler):
        self.params_dups = MlpModelConfig.update(self.params_dups,
                                                 self.model.step_size,
                                                 (next(sampler.iter[Input]), next(sampler.iter[Output])), self.rng)

    def predict(self, inputs: NDArray):
        return MlpModelConfig.forward(self.params_dups, inputs)

    def perform(self, data_src: Mapping[Port, NDArray], tgt: Port) -> NDArray:
        assert Input in data_src and tgt == Output
        return self.predict(data_src[Input])

    def perform_batch(self,
                      data_src: Mapping[Port, NDArray],
                      tgt: FrozenSet[Port]) -> Mapping[Port, NDArray]:
        assert Input in data_src and len(tgt) == 1 and next(iter(tgt)) == Output
        return {Output: self.predict(data_src[Input])}


# noinspection PyTypeChecker
# Because Pycharm sucks
mnist_in_flattened = MnistConfigIn(is_float=True, is_flat=True).get_var()
mnist_out_1hot = MnistConfigOut(is_1hot=True).get_var()
port_query: DataQuery = {Input: mnist_in_flattened, Output: mnist_out_1hot}
data_config_ = MnistDataConfig(base_path=Path('/Data/torchvision/'), port_vars=port_query)
benchmark_config_ = BenchmarkConfig(metrics={Output: get_mean_acc(10)}, on=FixedTest)
# noinspection PyTypeChecker
# Because Pycharm sucks
model_ = MlpModelConfig(
    step_size=0.03,
    num_epochs=50,
    train_batch_size=32,
    layer_sizes=[784, 1280, 512, 256, 10],
    train_data_config=data_config_
).prepare()
# noinspection PyTypeChecker
# Because Pycharm sucks
z = benchmark_config_.bench(data_config_, model_)
print(z)
