from pprint import pprint

import jax.numpy as xp
from jax import tree_flatten, tree_map, jit, vmap
from jax._src.random import PRNGKey

from tests.jax_random_utils import init_weights
from tests.transformer import TransformerConfigs, TransformerEncoder

config = TransformerConfigs(
    n_tfe_layers=3,
    n_seq=8,
    n_heads=4,
    dim_model=12,
    dim_input=12,
    dropout_keep_rate=1,
    eps=0.00001,
    mlp_n_hidden=[17],
    mlp_activation="relu"
)

C, N, T = config.dim_input, 10, 7
x = xp.ones((C, N, T))
tfe = TransformerEncoder.make(config)
weights = init_weights(tfe.params)
_, struct = tree_flatten(weights)
pprint(tree_map(lambda x: x.shape, weights))
rng = PRNGKey(0)
result = vmap(jit(tfe.process), (None, 0, None))(weights, x, rng)
print(result.shape)
