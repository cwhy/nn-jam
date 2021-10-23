# Vit for mnist
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Tuple, List, Mapping, FrozenSet, Literal, Callable, Any, Optional, Dict, NamedTuple

import jax
from jax import jit, grad, tree_map, vmap, tree_flatten, random
from jax._src.random import PRNGKey
from jax import numpy as xp
from jax.scipy.special import logsumexp
from numpy import typing as npt
from numpy.typing import NDArray
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Input, Output, Port, DataConfig, DataPool, DataContent
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.metrics import get_mean_acc
from supervised_benchmarks.mnist.mnist import MnistDataConfig, FixedTrain, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.model_utils import Train, Probes
from supervised_benchmarks.numpy_utils import ordinal_from_1hot
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import MiniBatchSampler, FixedEpochSamplerConfig, FullBatchSamplerConfig
from tests.jax_activations import Activation
from tests.jax_random_utils import ArrayTree, RNGKey, init_weights
from tests.vit import Vit


@dataclass(frozen=True)
class MlpModelConfig:
    step_size: float
    num_epochs: int
    train_batch_size: int
    activation: Activation
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

    # noinspection PyTypeChecker
    # because pycharm sucks
    def prepare(self) -> Performer[NDArray]:
        y_dim = 10
        config_train = Vit(
            n_heads=8,
            dim_model=32,
            dropout_keep_rate=1,
            eps=0.00001,
            mlp_n_hidden=[50],
            mlp_activation=self.activation,
            pos_t=-1,
            hwc=(28, 28, 1),
            n_patches_side=4,
            mlp_n_hidden_patches=[64],
            n_tfe_layers=8,
            dim_output=1,
            dict_size_output=y_dim,
            input_keep_rate=0.8,
        )
        config_test = config_train._replace(
            dropout_keep_rate=1,
            input_keep_rate=1,
        )
        vit_train = Vit.make(config_train)
        vit_test = Vit.make(config_test)

        weights = init_weights(vit_train.params)
        leaves, tree_def = tree_flatten(weights)
        print(tree_def)

        def get_logits(params, vit_outs):
            logits = params['out_embedding']['dict'] @ vit_outs[:, 0]
            print(logits.shape)
            print("logits", logits)
            return logits - logsumexp(logits, keepdims=True)

        @jit
        def forward_test(params, inputs):
            inputs = xp.expand_dims(inputs, -1)
            print(inputs.shape)
            # pprint(tree_map(lambda x: "{:.2f}, {:.2f}".format(x.mean().item(), x.std().item()), params))
            outs, _, _ = vit_test.process(params, inputs, random.PRNGKey(0))
            return xp.exp(get_logits(params, outs))

        def forward_train(params, inputs, rng):
            inputs = xp.expand_dims(inputs, -1)
            print(inputs.shape)
            outs, _, _ = vit_train.process(params, inputs, rng)
            return get_logits(params, outs)

        @jit
        def _loss(params, batch, rng):
            inputs, targets = batch
            rngs = random.split(rng, len(inputs))
            preds = vmap(forward_train, (None, 0, 0))(params, inputs, rngs)
            return -xp.mean(xp.sum(preds * targets, axis=1))

        # def cosine(a, b):
        #     return jax.lax.dot(a, b) / xp.linalg.norm(a) / xp.linalg.norm(b)

        # def l2m(a, b, mask):
        #     return mask*xp.mean((a - b) * (a - b))

        # def _loss2_1(params, inputs, targets, rng):
        #     inputs = xp.expand_dims(inputs, -1)
        #     outs, bf, idx = vit_train.process(params, inputs, rng)
        #     y = params['out_embedding']['dict'][ordinal_from_1hot(targets), :]
        #     y = y / xp.linalg.norm(y)
        #     print("yyyyyyyyyyyyy", y.shape, y)
        #     l = vmap(l2m, (-1, -1, -1), -1)(outs, xp.c_[y, bf], idx)
        #     print("lllllllllllllll", l)
        #     return l

        # @jit
        # def _loss2(params, batch, rng):
        #     inputs, targets = batch
        #     loss = vmap(_loss2_1, (None, 0, 0, None), 0)(params, inputs, targets, rng)
        #     return math.sqrt(self.train_batch_size) * xp.mean(loss)
        loss = _loss

        @jit
        def update(params, step_size: float, batch, rng: RNGKey):
            key, rng = random.split(rng)
            grads = grad(loss)(params, batch, key)
            return tree_map(lambda x, dx: x - step_size * dx, params, grads), rng

        model = MlpModel(self, weights, PRNGKey(0), vmap(forward_test, (None, 0), 0), update, loss)

        # pool_dict = self.train_data_config.get_data()
        # _data = subset_all(pool_dict, FixedTest)
        # data = FullBatchSamplerConfig().get_sampler(_data)
        # z = model.predict(data.full_batch[Input])
        # print(z.mean(axis=1))

        Train(
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            # BenchmarkConfig(metrics={Output: get_mean_acc(10)}, on=FixedTrain),
            bench_configs=[
                BenchmarkConfig(metrics={Output: get_mean_acc(10)},
                                on=FixedTest,
                                sampler_config=FixedEpochSamplerConfig(512)),
                BenchmarkConfig(metrics={Output: get_mean_acc(10)},
                                on=FixedTrain,
                                sampler_config=FixedEpochSamplerConfig(512))],
            model=model,
            data_subset=FixedTrain,
            data_config=self.train_data_config,
        ).run_()
        return model


@dataclass
class MlpModel:
    model: MlpModelConfig
    weights: ArrayTree
    state: RNGKey
    forward_test: Callable[[ArrayTree, npt.NDArray], npt.NDArray]
    update: Callable[[ArrayTree, float, Any, RNGKey], Tuple[ArrayTree, RNGKey]]
    loss: Callable[[ArrayTree, Any, RNGKey], npt.NDArray]

    @property
    def probe(self) -> Dict[Probes, Callable[[Mapping[Port, DataPool[DataContent]]], None]]:
        def debug(pool):
            cfg = FullBatchSamplerConfig()
            sp1 = cfg.get_sampler(subset_all(pool, FixedTrain)).full_batch
            sp2 = cfg.get_sampler(subset_all(pool, FixedTest)).full_batch
            i1, o1 = sp1[Input][:1000, :, :], sp1[Output][:1000, :]
            i2, o2 = sp2[Input][:1000, :, :], sp2[Output][:1000, :]
            print("loss: ",
                  self.loss(self.weights, (i1, o1), PRNGKey(0)),
                  self.loss(self.weights, (i2, o2), PRNGKey(0)))
        return {
            # "after_epoch_": lambda: print(self.weights['layer_0']['b'].mean(), self.weights['layer_0']['w'].mean())
            "after_epoch_": debug
        }

    def update_(self, sampler: MiniBatchSampler):
        self.weights, self.state = self.update(self.weights,
                                               self.model.step_size,
                                               (next(sampler.iter[Input]), next(sampler.iter[Output])),
                                               self.state)

    def predict(self, inputs: NDArray):
        return self.forward_test(self.weights, inputs)

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
        Input: MnistConfigIn(is_float=True, is_flat=False).get_var(),
        Output: MnistConfigOut(is_1hot=True).get_var()
    })

benchmark_config_ = BenchmarkConfig(
    metrics={Output: get_mean_acc(10)},
    on=FixedTest)

# noinspection PyTypeChecker
# Because Pycharm sucks
model_ = MlpModelConfig(
    step_size=0.01,
    num_epochs=200,
    train_batch_size=128,
    train_data_config=data_config_,
    activation='tanh'
).prepare()
# noinspection PyTypeChecker
# Because Pycharm sucks
z = benchmark_config_.bench(data_config_, model_)
print(z)
