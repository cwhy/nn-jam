import jax.numpy as xp
from jax import Array
from jax.nn import softmax

from jax_make.component_protocol import Component, FixedPipeline
from jax_make.params import ArrayTreeMapping, get_arr
from jax_make.params import WeightParams


# May not be efficient!!!


def attention(weights: ArrayTreeMapping, x: Array) -> Array:
    q, k, v = get_arr(weights, 'q'), get_arr(weights, 'k'), get_arr(weights, 'v')
    return softmax(x @ q @ (x @ k).T) @ v


# TODO masked attention
def causal_attention(weights: ArrayTreeMapping, x: Array) -> Array:
    q, k, v = get_arr(weights, 'q'), get_arr(weights, 'k'), get_arr(weights, 'v')
    attn_logits = x @ q @ (x @ k).T
    mask = xp.tril(xp.ones((x.shape[0], x.shape[0])))
    attn = softmax(xp.where(mask == 0, -xp.inf, attn_logits), axis=-1)
    return attn @ x @ v


def attention_component(
        head_size: int,  # k
        dim_input: int,  # x
        fixed_pipeline: FixedPipeline
) -> Component:
    return Component.from_fixed_pipeline(
        {"k": WeightParams(shape=(dim_input, head_size)),
         "q": WeightParams(shape=(dim_input, head_size)),
         "v": WeightParams(shape=(dim_input, head_size))},
        fixed_pipeline
    )
