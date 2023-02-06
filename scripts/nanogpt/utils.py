import jax
from jax import Array
import jax.numpy as xp

from jax_make.component_protocol import FixedPipeline


def infinite_jax_keys(seed: int):
    init_key = jax.random.PRNGKey(seed)
    while True:
        init_key, key = jax.random.split(init_key)
        yield key


def flatten_token(array: Array) -> Array:
    return xp.concatenate(array)


def batch_fy(fixed_pipeline: FixedPipeline) -> FixedPipeline:
    return jax.vmap(fixed_pipeline, in_axes=(None, 0), out_axes=0)
