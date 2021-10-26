from pprint import pprint

import jax.numpy as xp
from jax import jit, make_jaxpr, vmap, tree_map
from jax._src.random import PRNGKey

from jax_make.einops_utils import mix, MixWeights
from jax_make.jax_modules.dropout import Dropout
from jax_make.jax_modules.positional_encoding import PositionalEncoding
from jax_make.params import init_weights
from jax_make.jax_modules.multi_head_attn import SelfMultiHeadAttn

b, c, x, x1 = 4, 5, 2, 3
k = mix("b x c -> b x1 c", weight_shape='x x1', bias_shape='x1', x=x, x1=x1)
print(k)
print("finished jit")
weights = MixWeights(w=xp.ones(k.weight_shape), b=xp.ones(k.bias_shape))
k = k.make().process(weights, xp.ones((b, x, c)), PRNGKey(0))
print(k.shape)

H, T, K, C, N = 4, 5, 8, 10, 3
inputs = xp.ones((C, N, T))

mha_config = SelfMultiHeadAttn(
    n_heads=H,
    dim_model=K,
    dim_input=C)
mha2 = SelfMultiHeadAttn.make(mha_config)
# noinspection PyTypeChecker
weights2 = init_weights(mha2.params)
jit2 = jit(vmap(mha2.process, (None, 1, None), 1))

out_expr2 = make_jaxpr(jit2)(weights2, inputs, PRNGKey(1))
print(len(repr(out_expr2)))
out2 = jit(jit2)(weights2, inputs, PRNGKey(1))
print(out2.shape)

dropout = Dropout.make(Dropout(0.8))
print(dropout.params)
# noinspection PyTypeChecker


pos_encode = PositionalEncoding.make(PositionalEncoding(
    input_shape=(2, 3, 5, 7),
    input_channels=4,
    output_channels=4,
    dim_encoding=4,
    positional_encode_strategy='dot'
))
inputs = xp.ones((4, 2, 3, 5, 7))
weights = init_weights(pos_encode.params)
pprint(tree_map(lambda _x: _x.shape, weights))
out = pos_encode.fixed_process(weights, inputs)
print(out.shape)
