from __future__ import annotations

from pathlib import Path
from pprint import pprint

import jax.numpy as xp
from jax import tree_map, vmap, jit
from jax._src.random import PRNGKey

from jax_make.params import make_weights
from jax_make.transformer import TransformerEncoderConfigs
from jax_make.vit import Vit, VitConfigs
from supervised_benchmarks.mnist.mnist import get_mnist_

config = Vit(
    n_heads=4,
    dim_model=12,
    dropout_keep_rate=1.,
    eps=0.00001,
    mlp_n_hidden=[24],
    mlp_activation="relu",
    pos_t=-1,
    n_tfe_layers=4,
    universal=True,
    dict_init_scale=1,

    hwc=(28, 28, 1),
    n_patches_side=7,
    mlp_n_hidden_patches=[12],
    dim_output=1,
    dict_size_output=5,
    input_keep_rate=1,
    pos_init_scale=0.001
)
base_path = Path('/Data/torchvision/')
array_dict = get_mnist_(base_path)
l: TransformerEncoderConfigs = config
m: Vit = config
n: VitConfigs = config

inputs = xp.expand_dims(array_dict['train.images'][0:10], -1).astype(xp.float_)
print(inputs.shape)

tf = Vit.make(config)
print(tf.processes)
print(next(iter(tf.processes.values())))
print(type(next(iter(tf.processes.values()))))
weights = make_weights(tf.weight_params)
pprint(tree_map(lambda _x: _x.shape, weights))
rng = PRNGKey(0)
print(type(inputs))
# result = vmap(tf.pipeline, (None, 0, None), 0)(weights, inputs, rng)
result = vmap(jit(tf.pipeline), (None, 0, None), 0)(weights, inputs, rng)
print(result)
