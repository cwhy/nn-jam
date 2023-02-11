from pprint import pprint

import jax.numpy as xp
from jax import jit, make_jaxpr, vmap, tree_map
from jax._src.random import PRNGKey

from jax_make.components.dropout import Dropout
from jax_make.components.tensor_positional_encoding import TensorPositionalEncoding
from jax_make.params import make_weights
from jax_make.components.multi_head_attn import SelfMultiHeadAttn


H, T, K, C, N = 4, 5, 8, 10, 3
inputs = xp.ones((C, N, T))

mha_config = SelfMultiHeadAttn(
    n_heads=H,
    dim_model=K,
    dim_input=C)
mha2 = SelfMultiHeadAttn.make(mha_config)
# noinspection PyTypeChecker
weights2 = make_weights(mha2.weight_params)
jit2 = jit(vmap(mha2.pipeline, (None, 1, None), 1))

out_expr2 = make_jaxpr(jit2)(weights2, inputs, PRNGKey(1))
print(len(repr(out_expr2)))
out2 = jit(jit2)(weights2, inputs, PRNGKey(1))
print(out2.shape)

dropout = Dropout.make(Dropout(0.8))
print(dropout.weight_params)
# noinspection PyTypeChecker


pos_encode = TensorPositionalEncoding.make(TensorPositionalEncoding(
    input_shape=(2, 3, 5, 7),
    input_channels=4,
    output_channels=4,
    dim_encoding=4,
    positional_encode_strategy='sum'
))
inputs = xp.ones((4, 2, 3, 5, 7))
# noinspection PyTypeChecker
weights = make_weights(pos_encode.weight_params)
pprint(tree_map(lambda _x: _x.shape, weights))
out = pos_encode.fixed_pipeline(weights, inputs)
print(out.shape)
