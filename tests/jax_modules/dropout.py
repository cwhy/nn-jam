from typing import NamedTuple, Protocol, Tuple

import jax.numpy as xp
import numpy.typing as npt

from tests.jax_components import Component
from tests.jax_random_utils import RandomParams, ArrayTree


class DropoutConfigs(Protocol):
    dropout_keep_rate: float
    single_input_shape: Tuple[int, ...]


class Dropout(NamedTuple):
    dropout_keep_rate: float
    single_input_shape: Tuple[int, ...]

    @staticmethod
    def make(config: DropoutConfigs) -> Component:
        components = {
            'dropout_keep': RandomParams(shape=config.single_input_shape,
                                         init="dropout",
                                         scale=config.dropout_keep_rate),
        }

        def _fn(weights: ArrayTree, x: npt.NDArray) -> npt.NDArray:
            print(weights['dropout_keep'][0, :10])
            return xp.where(weights['dropout_keep'], x / config.dropout_keep_rate, 0)

        return Component(components, _fn)

