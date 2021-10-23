from typing import NamedTuple, Protocol

import jax.numpy as xp
from jax import random
from numpy.typing import NDArray

from tests.jax_components import Component, X
from tests.jax_random_utils import ArrayTree, RNGKey


class DropoutConfigs(Protocol):
    dropout_keep_rate: float


class Dropout(NamedTuple):
    dropout_keep_rate: float

    @staticmethod
    def make(config: DropoutConfigs) -> Component:
        def _fn(_: ArrayTree, x: NDArray, rng_key: RNGKey) -> NDArray:
            if 0 < config.dropout_keep_rate < 1:
                rd = random.bernoulli(rng_key, config.dropout_keep_rate, x.shape)
                return xp.where(rd, x / config.dropout_keep_rate, 0)
            elif config.dropout_keep_rate == 1:
                return x
            else:
                raise Exception(f"Dropout rate should be in (0, 1], found {config.dropout_keep_rate}")

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline({}, _fn)

