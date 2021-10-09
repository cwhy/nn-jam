from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Mapping, FrozenSet, Literal, Callable, Any, Optional, Dict, NamedTuple

import jax.numpy as jnp
from jax import jit, grad, tree_map, vmap, tree_flatten, random
from jax._src.random import PRNGKey
from jax.scipy.special import logsumexp
from numpy import typing as npt
from numpy.typing import NDArray
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Input, Output, Port, DataConfig
from supervised_benchmarks.metrics import get_mean_acc
from supervised_benchmarks.mnist.mnist import MnistDataConfig, FixedTrain, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.model_utils import Train, Probes
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import MiniBatchSampler
from tests.jax_activations import Activation
from tests.jax_modules.mlp import Mlp
from tests.jax_random_utils import ArrayTree, RNGKey, init_weights


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
        mlp_config_train = Mlp(
            n_in=28 * 28,
            n_hidden=self.layer_sizes[1:-1],
            n_out=self.layer_sizes[-1],
            activation=self.activation,
            dropout_keep_rate=0.8,
        )
        mlp_config_test = mlp_config_train._replace(
            dropout_keep_rate=1
        )
        mlp_train = Mlp.make(mlp_config_train)
        mlp_test = Mlp.make(mlp_config_test)

        weights = init_weights(mlp_train.params)
        leaves, tree_def = tree_flatten(weights)
        print(tree_def)

        @jit
        def forward_test(params, inputs):
            logits = mlp_test.process(params, inputs, random.PRNGKey(0))
            return logits - logsumexp(logits, keepdims=True)

        def forward_train(params, inputs, rng):
            logits = mlp_train.process(params, inputs, rng)
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

        model = MlpModel(self, weights, PRNGKey(0), forward_test, update)
        Train(
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            bench_configs=[BenchmarkConfig(metrics={Output: get_mean_acc(10)}, on=FixedTrain),
                           BenchmarkConfig(metrics={Output: get_mean_acc(10)}, on=FixedTest)],
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
        Input: MnistConfigIn(is_float=True, is_flat=True).get_var(),
        Output: MnistConfigOut(is_1hot=True).get_var()
    })

benchmark_config_ = BenchmarkConfig(
    metrics={Output: get_mean_acc(10)},
    on=FixedTest)

# noinspection PyTypeChecker
# Because Pycharm sucks
model_ = MlpModelConfig(
    step_size=0.03,
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
