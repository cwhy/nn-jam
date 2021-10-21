from typing import NamedTuple, Protocol, Tuple

import jax.numpy as xp
import numpy.typing as npt
from jax import random

from tests.jax_components import Component
from tests.jax_random_utils import ArrayTree, RNGKey


class DropoutConfigs(Protocol):
    dropout_keep_rate: float


class Dropout(NamedTuple):
    dropout_keep_rate: float

    @staticmethod
    def make(config: DropoutConfigs) -> Component:
        def _fn(_: ArrayTree, x: npt.NDArray, rng_key: RNGKey) -> npt.NDArray:
            if 0 < config.dropout_keep_rate < 1:
                rd = random.bernoulli(rng_key, config.dropout_keep_rate, x.shape)
                return xp.where(rd, x / config.dropout_keep_rate, 0)
            elif config.dropout_keep_rate == 1:
                return x
            else:
                raise Exception(f"Dropout rate should be in (0, 1], found {config.dropout_keep_rate}")

        return Component({}, _fn)

