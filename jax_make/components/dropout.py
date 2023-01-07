from typing import NamedTuple, Protocol

import jax.numpy as xp
from jax import random
from numpy.typing import NDArray

from jax_make.component_protocol import Component, X
from jax_make.params import ArrayTree, RNGKey, ArrayTreeMapping
from abc import abstractmethod


class DropoutConfigs(Protocol):
    # add a property with the same name
    @property
    @abstractmethod
    def dropout_keep_rate(self) -> float:
        ...


class Dropout(NamedTuple):
    dropout_keep_rate: float

    @staticmethod
    def make(config: DropoutConfigs) -> Component:
        def _fn(weights: ArrayTreeMapping, x: NDArray, rng: RNGKey) -> NDArray:
            if 0 < config.dropout_keep_rate < 1:
                rd = random.bernoulli(rng, config.dropout_keep_rate, x.shape)
                return xp.where(rd, x / config.dropout_keep_rate, 0)
            elif config.dropout_keep_rate == 1:
                return x
            else:
                raise Exception(f"Dropout rate should be in (0, 1], found {config.dropout_keep_rate}")

        return Component.from_pipeline({}, _fn)
