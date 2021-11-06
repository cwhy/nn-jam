# Test anynet on mnist
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from random import randint
from typing import Tuple, Mapping, FrozenSet, Literal, Callable, Any, Dict

import optax as optax
from jax import jit, grad, tree_map, vmap, random, tree_leaves
from jax import numpy as xp
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from numpy import typing as npt
from numpy.typing import NDArray
from optax._src.alias import adamw
from optax._src.base import OptState
from pynng import Pub0
from variable_protocols.protocols import Variable
from variable_protocols.variables import one_hot, var_tensor, gaussian, dim, var_scalar, ordinal

from jax_make.anynet import AnyNet, inference_ports, QUERY_MASK, FLOAT_OFFSET, VALUE_SYMBOL
from jax_make.component_protocol import make_ports
from jax_make.params import ArrayTree, RNGKey, make_weights
from jax_make.vit import VitReconstruct
from stage.protocol import Stage
from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Input, Output, Port, DataConfig, DataPool, DataContent
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.metrics import get_pair_metric
from supervised_benchmarks.mnist.mnist import MnistDataConfig, FixedTrain, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.model_utils import Train, Probes
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import MiniBatchSampler, FixedEpochSamplerConfig, FullBatchSamplerConfig


@dataclass(frozen=True)
class MlpModelConfig:
    dim_model: int
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
        weight_decay = 0.0001
        len_x = 28 * 28
        len_y = 1
        confit_any_net = AnyNet(
            universal=True,
            n_heads=8,
            dim_model=self.dim_model,
            dropout_keep_rate=1,
            eps=eps,
            mlp_n_hidden=[128],
            mlp_activation='gelu',
            pos_t=-1,
            n_tfe_layers=3,
            input_keep_rate=0.5,

            init_embed_scale=0.01,
            n_symbols=10,
            n_positions=len_x + len_y,
            max_inputs=len_x + len_y
        )
        config_test = confit_any_net._replace(
            dropout_keep_rate=1,
            input_keep_rate=1,
        )
        net_train = AnyNet.make(confit_any_net)
        net_test = AnyNet.make(config_test)
        pprint(tree_map(lambda x: x, net_train.weight_params))
        weights = make_weights(net_train.weight_params)
        pprint(tree_map(lambda x: "{:.2f}, {:.2f}".format(x.mean().item(), x.std().item()), weights))
        print("params count:", sum(x.size for x in tree_leaves(weights)))
        # leaves, tree_def = tree_flatten(weights)
        # print(tree_def)
        mask_full = xp.ones(len_x + len_y)
        pos_sequence_all = xp.arange(len_x + len_y)
        empty_values_ask = xp.zeros(len_x + len_y)
        float_offsets = xp.ones(len_x, dtype=int) * FLOAT_OFFSET

        def forward_test(params, x):
            inputs = {Input: xp.c_[x, QUERY_MASK], 'input_pos': pos_sequence_all, 'value': empty_values_ask}
            outs = net_test.processes[inference_ports](params, inputs, random.PRNGKey(0))['symbol']
            return outs[-1]  # -1, Not mask!!

        def get_batch_loss_process(param, x, y, rng):
            inputs = {Input: xp.r_[float_offsets, y],
                      'mask': mask_full,
                      'input_pos': pos_sequence_all,
                      'value': xp.r_[x, VALUE_SYMBOL]}
            process = net_train.processes[make_ports((Input, 'mask', 'input_pos', 'value'), 'loss')]
            return process(param, inputs, rng)

        def forward_train(params, batch, rng):
            x, y = batch[Input], batch[Output]
            rngs = random.split(rng, batch[Output].shape[0])
            return vmap(get_batch_loss_process, (None, 0, 0, 0))(params, x, y, rngs)

        def loss(params, batch, rng):
            return forward_train(params, batch, rng)['loss'].mean()

        def update_adam(params, batch, _state: Tuple[RNGKey, OptState]):
            rng, _opt_state = _state
            key, rng = random.split(rng)
            grads = grad(loss)(params, batch, key)
            updates, _opt_state = optimiser.update(grads, _opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, (rng, _opt_state)

        optimiser = adamw(self.step_size, 0.9, 0.98, eps, weight_decay)
        # schedule_fn = polynomial_schedule(
        #     init_value=-self.step_size/10, end_value=-self.step_size, power=1, transition_steps=5000)
        # optimiser = chain(
        #     scale_by_adam(eps=1e-4),
        #     scale_by_schedule(schedule_fn))
        opt_state: OptState = optimiser.init(weights)

        with Pub0(dial='tcp://127.0.0.1:54323') as pub:
            stage = Stage(pub)
            state = (PRNGKey(0), opt_state)
            model = MlpModel(self, weights, state,
                             vmap(jit(forward_test), (None, 0), 0), jit(update_adam), jit(forward_train),
                             train_stage=stage)
            Train(
                num_epochs=self.num_epochs,
                batch_size=self.train_batch_size,
                bench_configs=[
                    BenchmarkConfig(metrics={Output: get_pair_metric('mean_acc', var_scalar(ordinal(y_dim)))},
                                    on=FixedTest,
                                    sampler_config=FixedEpochSamplerConfig(512)),
                    #     BenchmarkConfig(metrics={Output: get_pair_metric('mean_acc', var_scalar(ordinal(y_dim)))},
                    #                     on=FixedTrain,
                    #                     sampler_config=FixedEpochSamplerConfig(512)),
                ],
                model=model,
                data_subset=FixedTrain,
                data_config=self.train_data_config,
            ).run_()
        return model


@dataclass
class MlpModel:
    model: MlpModelConfig
    weights: ArrayTree
    state: Tuple[RNGKey, OptState]
    forward_test: Callable[[ArrayTree, npt.NDArray], npt.NDArray]
    update: Callable[[ArrayTree, Any, Tuple[RNGKey, OptState]], Tuple[ArrayTree, Tuple[RNGKey, OptState]]]
    forward_train: Callable[[ArrayTree, Any, RNGKey], ArrayTree]
    train_stage: Stage

    @property
    def probe(self) -> Dict[Probes, Callable[[Mapping[Port, DataPool[DataContent]]], None]]:
        def debug(pool):
            cfg = FullBatchSamplerConfig()
            sp1 = cfg.get_sampler(subset_all(pool, FixedTrain)).full_batch
            sp2 = cfg.get_sampler(subset_all(pool, FixedTest)).full_batch

            i1, o1 = sp1[Input][:10, :], sp1[Output][:10]
            i2, o2 = sp2[Input][:10, :], sp2[Output][:10]
            forwarded_train = self.forward_train(self.weights, {Input: i1, Output: o1}, self.state[0])
            forwarded_test = self.forward_train(self.weights, {Input: i2, Output: o2}, self.state[0])
            loss_tr = forwarded_train['loss'].mean()
            print("loss: ",
                  loss_tr,
                  forwarded_test['loss'].mean())
            pos_encode = self.weights['positional_embedding']['dict']
            print(pos_encode.std())
            corr = xp.corrcoef(pos_encode.T)
            corr2 = xp.corrcoef(self.weights['symbol_embedding']['dict'])
            self.train_stage.socket.send(pickle.dumps(dict(x=corr, y=corr2,
                                                           loss_tr=loss_tr)))

        return {
            # "after_epoch_": lambda: print(self.weights['layer_0']['b'].mean(), self.weights['layer_0']['w'].mean())
            "before_epoch_": debug
        }

    def update_(self, sampler: MiniBatchSampler):
        self.weights, self.state = self.update(self.weights,
                                               {Input: next(sampler.iter[Input]), Output: next(sampler.iter[Output])},
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
        Output: MnistConfigOut(is_1hot=False).get_var()
    })

# noinspection PyTypeChecker
# Because Pycharm sucks
benchmark_config_ = BenchmarkConfig(
    metrics={Output: get_pair_metric('mean_acc', data_config_.port_vars[Output])},
    on=FixedTest)

# noinspection PyTypeChecker
# Because Pycharm sucks
model_ = MlpModelConfig(
    dim_model=32,
    step_size=0.01,
    num_epochs=20000,
    train_batch_size=64,
    train_data_config=data_config_,
).prepare()
# noinspection PyTypeChecker
# Because Pycharm sucks
z = benchmark_config_.bench(data_config_, model_)
print(z)
