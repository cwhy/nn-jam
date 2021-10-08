from typing import NamedTuple, List, Protocol, Tuple

import jax.numpy as xp
import numpy.typing as npt

from tests.jax_activations import Activation, get_activation
from tests.jax_modules.dropout import Dropout
from tests.jax_protocols import Component, Weights, WeightParams


class MlpConfigs(Protocol):
    input_shape: Tuple[int, ...]
    on_axis: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float


class Mlp(NamedTuple):
    input_shape: Tuple[int, ...]
    on_axis: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float

    @staticmethod
    def make(config: MlpConfigs) -> Component:
        n_in = config.input_shape[config.on_axis]
        u_ins = [n_in] + config.n_hidden
        u_outs = config.n_hidden + [config.n_out]
        components = {
            f"layer_{i}": Component({"w": WeightParams(shape=(_in, _out)),
                                     "b": WeightParams(shape=(_out,), init=0)},
                                    lambda params, x: x @ params['w'] + params['b']
                                    )
            for i, (_in, _out) in enumerate(zip(u_ins, u_outs))
        }
        if config.dropout_keep_rate != 1:
            for i, _out in enumerate(u_outs):
                new_shape = tuple(n if n != config.on_axis else _out
                                  for n in enumerate(config.input_shape))
                components[f'dropout_{i}'] = Dropout.make(Dropout(
                    dropout_keep_rate=config.dropout_keep_rate,
                    input_shape=new_shape
                ))

        def _fn(weights: Weights, x: npt.NDArray) -> npt.NDArray:
            flow_ = xp.moveaxis(x, config.on_axis, -1)
            activation = get_activation(config.activation)
            for layer_i in range(len(config.n_hidden)):
                layer_name = f"layer_{layer_i}"
                flow_ = components[layer_name].process(weights[layer_name], flow_)
                flow_ = activation(flow_)
                if config.dropout_keep_rate != 1:
                    flow_ = components[f"dropout_{layer_i}"]
            output_layer = f"layer_{len(config.n_hidden)}"
            flow_ = components[output_layer].process(weights[output_layer], flow_)
            flow_ = xp.moveaxis(flow_, -1, config.on_axis)
            return flow_

        return Component({k: v.weight_params for k, v in components.items()}, _fn)
