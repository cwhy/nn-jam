from pprint import pprint

import jax.numpy as xp
from jax import tree_flatten, tree_map, jit, vmap, random
from jax.random import PRNGKey

from jax_make.params import make_weights
from jax_make.transformer import Transformer, TransformerEncoder

config = Transformer(
    n_tfe_layers=3,
    n_seq=7,
    n_heads=4,
    dim_model=12,
    dim_input=12,
    dropout_keep_rate=1.,
    eps=0.00001,
    mlp_n_hidden=[17],
    mlp_activation="relu",
    dict_size=256,
    pos_t=-1
)

C, N, T = config.dim_input, 10, 7
x = xp.ones((C, N, T))
tfe = TransformerEncoder.make(config)
# noinspection PyTypeChecker
weights = make_weights(tfe.weight_params)
_, struct = tree_flatten(weights)
pprint(tree_map(lambda _x: _x.shape, weights))
rng = PRNGKey(0)
rng, key = random.split(rng)
result = vmap(jit(tfe.pipeline), (None, 1, None), 1)(weights, x, rng)
print(result.shape)


x_raw = random.randint(key, (N, T), 0, config.dict_size)
# noinspection PyTypeChecker
tf = Transformer.make(config)
# noinspection PyTypeChecker
weights = make_weights(tf.weight_params)
pprint(tree_map(lambda _x: _x.shape, weights))
result = vmap(jit(tf.pipeline), (None, 0, None), 0)(weights, x_raw, rng)
print(result)
print(result.shape)
