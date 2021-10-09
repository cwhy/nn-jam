from typing import NamedTuple, List, Protocol

import numpy.typing as npt
from jax import random

from tests.jax_activations import Activation, get_activation
from tests.jax_modules.dropout import Dropout
from tests.jax_components import Component
from tests.jax_paramed_functions import linear
from tests.jax_random_utils import WeightParams, ArrayTree, RNGKey


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
        components = {
            f"layer_{i}": Component.from_fixed_process(
                {"w": WeightParams(shape=(_in, _out)),
                 "b": WeightParams(shape=(_out,), init=0)},
                linear
            )
            for i, (_in, _out) in enumerate(zip(u_ins, u_outs))
        }
        if config.dropout_keep_rate != 1:
            for i, _out in enumerate(u_outs):
                components[f'dropout_{i}'] = Dropout.make(config)

        def _fn(weights: ArrayTree, flow_: npt.NDArray, rng: RNGKey) -> npt.NDArray:
            activation = get_activation(config.activation)
            n_layers = len(config.n_hidden)
            keys = random.split(rng, n_layers)
            for layer_i in range(n_layers):
                layer_name = f"layer_{layer_i}"
                flow_ = components[layer_name].fixed_process(weights[layer_name], flow_)
                flow_ = activation(flow_)
                if config.dropout_keep_rate != 1:
                    flow_ = components[f"dropout_{layer_i}"].process(weights[layer_name], flow_, keys[layer_i])
            output_layer = f"layer_{len(config.n_hidden)}"
            flow_ = components[output_layer].fixed_process(weights[output_layer], flow_)
            return flow_

        return Component({k: v.params for k, v in components.items()}, _fn)
