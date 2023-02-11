from functools import partial
from typing import Dict, Callable, Tuple

import jax
import jax.numpy as xp
import optax
from jax import Array

from jax_make.component_protocol import FixedPipeline, FixedProcess
from jax_make.params import ArrayTreeMapping


def infinite_jax_keys(seed: int):
    init_key = jax.random.PRNGKey(seed)
    while True:
        init_key, key = jax.random.split(init_key)
        yield key


def flatten_token(array: Array) -> Array:
    return xp.concatenate(array)


def batch_fy(fixed_pipeline: FixedPipeline) -> FixedPipeline:
    return jax.vmap(fixed_pipeline, in_axes=(None, 0), out_axes=0)


@partial(jax.jit, static_argnums=(0, 1))
def jax_calc_updates(
        optimizer: optax.GradientTransformation,
        loss_fn: Callable[[ArrayTreeMapping, Dict[str, Array]], Array],
        weights: ArrayTreeMapping,
        batch: Dict[str, Array],
        opt_state: optax.OptState
) -> Tuple[ArrayTreeMapping, optax.OptState]:
    grads = jax.grad(loss_fn)(weights, batch)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    return optax.apply_updates(weights, updates), opt_state
