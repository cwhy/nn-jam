# Vit for mnist
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Tuple, List, Mapping, FrozenSet, Literal, Callable, Any, Optional, Dict, NamedTuple

import jax.numpy as jnp
from jax import jit, grad, tree_map, vmap, tree_flatten, random
from jax._src.random import PRNGKey
from jax import numpy as xp
from jax.scipy.special import logsumexp
from numpy import typing as npt
from numpy.typing import NDArray
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Input, Output, Port, DataConfig
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.metrics import get_mean_acc
from supervised_benchmarks.mnist.mnist import MnistDataConfig, FixedTrain, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.model_utils import Train, Probes
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import MiniBatchSampler, FixedEpochSamplerConfig, FullBatchSamplerConfig
from tests.jax_activations import Activation
from tests.jax_random_utils import ArrayTree, RNGKey, init_weights
from tests.transformer import Vit


@dataclass(frozen=True)
class MlpModelConfig:
    step_size: float
    num_epochs: int
    train_batch_size: int
    layer_sizes: List[int]
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
            n_heads=4,
            dim_model=12,
            dropout_keep_rate=1.,
            eps=0.00001,
            mlp_n_hidden=[24],
            mlp_activation="relu",
            pos_t=-1,
            hwc=(28, 28, 1),
            n_patches_side=4,
            mlp_n_hidden_patches=[12],
            n_tfe_layers=4,
            dim_output=1,
            dict_size_output=y_dim
        )
        config_test = config_train._replace(
            dropout_keep_rate=1
        )
        vit_train = Vit.make(config_train)
        vit_test = Vit.make(config_test)

        weights = init_weights(vit_train.params)
        leaves, tree_def = tree_flatten(weights)
        print(tree_def)

        @jit
        def forward_test(params, inputs):
            inputs = xp.expand_dims(inputs, -1)
            print(inputs.shape)
            # pprint(tree_map(lambda x: "{:.2f}, {:.2f}".format(x.mean().item(), x.std().item()), params))
            outs = vit_test.process(params, inputs, random.PRNGKey(0))[:, -1]
            logits = vmap(xp.dot, (0, 1))(params['out_embedding']['dict'], outs)
            print(logits.shape)
            print("logits", logits)
            return xp.exp(logits - logsumexp(logits, keepdims=True))

        def forward_train(params, inputs, rng):
            inputs = xp.expand_dims(inputs, -1)
            print(inputs.shape)
            outs = vit_train.process(params, inputs, rng)[:, -1]
            logits = vmap(xp.dot, (0, 1))(params['out_embedding']['dict'], outs)
            print(logits.shape)
            return logits - logsumexp(logits, keepdims=True)

        def _loss(params, batch, rng):
            inputs, targets = batch
            rngs = random.split(rng, len(inputs))
            preds = vmap(forward_train, (None, 0, 0))(params, inputs, rngs)
            return -jnp.mean(jnp.sum(preds * targets, axis=1))

        @jit
        def update(params, step_size: float, batch, rng: RNGKey):
            key, rng = random.split(rng)
            grads = grad(_loss)(params, batch, key)
            return tree_map(lambda x, dx: x - step_size * dx, params, grads), rng

        model = MlpModel(self, weights, PRNGKey(0), vmap(forward_test, (None, 0), 0), update)

        pool_dict = self.train_data_config.get_data()
        _data = subset_all(pool_dict, FixedTest)
        data = FullBatchSamplerConfig().get_sampler(_data)
        z = model.predict(data.full_batch[Input])
        print(z.mean(axis=1))

        Train(
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            # BenchmarkConfig(metrics={Output: get_mean_acc(10)}, on=FixedTrain),
            bench_configs=[
                           BenchmarkConfig(metrics={Output: get_mean_acc(10)},
                                           on=FixedTrain,
                                           sampler_config=FixedEpochSamplerConfig(32))],
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

    @property
    def probe(self) -> Dict[Probes, Callable[[], None]]:
        return {
            # "after_epoch_": lambda: print(self.weights['layer_0']['b'].mean(), self.weights['layer_0']['w'].mean())
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
    step_size=0.003,
    num_epochs=30,
    train_batch_size=32,
    layer_sizes=[784, 512, 256, 10],
    train_data_config=data_config_,
    activation='relu'
).prepare()
# noinspection PyTypeChecker
# Because Pycharm sucks
z = benchmark_config_.bench(data_config_, model_)
print(z)
