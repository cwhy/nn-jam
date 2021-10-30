from typing import NamedTuple, List, Protocol

import numpy.typing as npt
from jax import random
from numpy.typing import NDArray

from jax_make.activations import Activation, get_activation
from jax_make.jax_modules.dropout import Dropout
from jax_make.components import Component, merge_params, X
from jax_make.jax_paramed_functions import linear
from jax_make.params import WeightParams, ArrayTree, RNGKey


class MlpConfigs(Protocol):
    n_in: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float


class Mlp(NamedTuple):
    n_in: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float

    @staticmethod
    def make(config: MlpConfigs) -> Component:
        u_ins = [config.n_in] + config.n_hidden
        u_outs = config.n_hidden + [config.n_out]
        # noinspection PyTypeChecker
        # Because pycharm sucks
        components = {
            f"layer_{i}": Component.from_fixed_pipeline(
                {"w": WeightParams(shape=(_in, _out)),
                 "b": WeightParams(shape=(_out,), init=0)},
                linear
            )
            for i, (_in, _out) in enumerate(zip(u_ins, u_outs))
        }
        if config.dropout_keep_rate != 1:
            for i, _out in enumerate(u_outs):
                components[f'dropout_{i}'] = Dropout.make(config)

        # n_in -> n_out
        def _fn(weights: ArrayTree, flow_: NDArray, rng: RNGKey) -> NDArray:
            activation = get_activation(config.activation)
            n_layers = len(config.n_hidden)
            keys = random.split(rng, n_layers)
            for layer_i in range(n_layers):
                layer_name = f"layer_{layer_i}"
                flow_ = components[layer_name].fixed_pipeline(weights[layer_name], flow_)
                flow_ = activation(flow_)
                if config.dropout_keep_rate != 1:
                    flow_ = components[f"dropout_{layer_i}"].process(weights[layer_name], flow_, keys[layer_i])
            output_layer = f"layer_{len(config.n_hidden)}"
            flow_ = components[output_layer].fixed_pipeline(weights[output_layer], flow_)
            return flow_

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline(merge_params(components), _fn)
