# Vit for mnist with reconstruction
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Mapping, FrozenSet, Literal, Callable, Any, Dict

from jax import jit, grad, tree_map, vmap, tree_flatten, random
from jax import numpy as xp
from jax._src.random import PRNGKey
from jax.scipy.special import logsumexp
from numpy import typing as npt
from numpy.typing import NDArray
from pynng import Pair0
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar

from jax_make.jax_modules.positional_encoding import dot_product_encode
from jax_make.params import ArrayTree, RNGKey, init_weights
from jax_make.vit import Vit, VitReconstruct, get_reconstruct_loss
from stage.protocol import Stage
from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Input, Output, Port, DataConfig, DataPool, DataContent
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.metrics import get_mean_acc
from supervised_benchmarks.mnist.mnist import MnistDataConfig, FixedTrain, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.model_utils import Train, Probes
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import MiniBatchSampler, FixedEpochSamplerConfig, FullBatchSamplerConfig


@dataclass(frozen=True)
class MlpModelConfig:
    step_size: float
    num_epochs: int
    train_batch_size: int
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
        eps = 0.00001
        config_vit = Vit(
            n_heads=8,
            dim_model=32,
            dropout_keep_rate=1,
            eps=eps,
            mlp_n_hidden=[50],
            mlp_activation='gelu',
            pos_t=-1,
            hwc=(28, 28, 1),
            n_patches_side=4,
            mlp_n_hidden_patches=[64],
            n_tfe_layers=8,
            dim_output=1,
            dict_size_output=y_dim,
            input_keep_rate=0.8,
        )
        config_test = config_vit._replace(
            dropout_keep_rate=1,
            input_keep_rate=1,
        )
        vit_train = VitReconstruct.make(config_vit)
        vit_test = VitReconstruct.make(config_test)

        weights = init_weights(vit_train.params)
        leaves, tree_def = tree_flatten(weights)
        print(tree_def)

        def get_logits(params, y_rec):
            logits = params['out_embedding']['dict'] @ y_rec
            print(logits.shape)
            print("logits", logits)
            return logits - logsumexp(logits, keepdims=True)

        @jit
        def forward_test(params, inputs):
            # pprint(tree_map(lambda x: "{:.2f}, {:.2f}".format(x.mean().item(), x.std().item()), params))
            outs = vit_test.process(params, inputs, random.PRNGKey(0))
            return xp.exp(get_logits(params, outs['y']))

        def forward_train(params, inputs, rng):
            print(inputs.shape)
            outs = vit_train.process(params, inputs, rng)
            return get_logits(params, outs)

        def _loss(params, batch, rng):
            rngs = random.split(rng, len(batch['X']))
            outs = vmap(forward_train, (None, 0, 0))(params, batch['X'], rngs)
            preds = vmap(get_reconstruct_loss(eps), [0]*5, 0)(outs['x'], batch[''], outs[''], outs[''])
            return -xp.mean()

        @jit
        def _loss_all(params, batch, rng):
            rngs = random.split(rng, len(batch['X']))
            return -xp.mean()

        loss = _loss

        @jit
        def update(params, step_size: float, batch, rng: RNGKey):
            key, rng = random.split(rng)
            print("batches: ", batch)
            grads = grad(loss)(params, batch, key)
            return tree_map(lambda x, dx: x - step_size * dx, params, grads), rng

        with Pair0(dial='tcp://127.0.0.1:54322') as socket:
            stage = Stage(socket)
            model = MlpModel(self, weights, PRNGKey(0), vmap(forward_test, (None, 0), 0), update, loss,
                             train_stage=stage)
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
    train_stage: Stage

    @property
    def probe(self) -> Dict[Probes, Callable[[Mapping[Port, DataPool[DataContent]]], None]]:
        def debug(pool):
            cfg = FullBatchSamplerConfig()
            sp1 = cfg.get_sampler(subset_all(pool, FixedTrain)).full_batch
            sp2 = cfg.get_sampler(subset_all(pool, FixedTest)).full_batch
            i1, o1 = sp1[Input][:1000, :, :], sp1[Output][:1000, :]
            i2, o2 = sp2[Input][:1000, :, :], sp2[Output][:1000, :]
            print("loss: ",
                  self.loss(self.weights, {"X": i1, "Y": o1}, PRNGKey(0)),
                  self.loss(self.weights, {"X": i2, "Y": o2}, PRNGKey(0)))
            pos_encode = dot_product_encode(self.weights['positional_encoding'], 3).reshape(32, -1)
            corr = xp.cov(pos_encode.T)
            # corr = (corr - corr.min()) / (corr.max() - corr.min()) * 255
            self.train_stage.socket.send(pickle.dumps(dict(x=corr)))

        return {
            # "after_epoch_": lambda: print(self.weights['layer_0']['b'].mean(), self.weights['layer_0']['w'].mean())
            "after_epoch_": debug
        }

    def update_(self, sampler: MiniBatchSampler):
        self.weights, self.state = self.update(self.weights,
                                               self.model.step_size,
                                               {'X': next(sampler.iter[Input]), 'Y': next(sampler.iter[Output])},
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
).prepare()
# noinspection PyTypeChecker
# Because Pycharm sucks
z = benchmark_config_.bench(data_config_, model_)
print(z)
