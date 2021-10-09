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
            rd = random.bernoulli(rng_key, config.dropout_keep_rate, x.shape)
            return xp.where(rd, x / config.dropout_keep_rate, 0)

        return Component({}, _fn)

