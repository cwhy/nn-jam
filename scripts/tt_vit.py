from __future__ import annotations

from pathlib import Path
from pprint import pprint

import jax.numpy as xp
from jax import tree_map, jit, vmap
from jax._src.random import PRNGKey

from supervised_benchmarks.dataset_protocols import DataSubset
from supervised_benchmarks.ports import Input, Output
from supervised_benchmarks.mnist.mnist import MnistDataConfig, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from jax_make.params import make_weights
from jax_make.vit import Vit

config = Vit(
    n_heads=4,
    dim_model=12,
    dropout_keep_rate=1.,
    eps=0.00001,
    mlp_n_hidden=[24],
    mlp_activation="relu",
    pos_t=-1,
    hwc=(28, 28, 1),
    n_patches_side=7,
    mlp_n_hidden_patches=[12],
    n_tfe_layers=4,
    dim_output=1,
    dict_size_output=5,
    input_keep_rate=1
)

data_config_ = MnistDataConfig(
    base_path=Path('/Data/torchvision/'),
    port_vars={
        Input: MnistConfigIn(is_float=True, is_flat=False).get_var(),
        Output: MnistConfigOut(is_1hot=True).get_var()
    })
td: DataSubset = data_config_.get_data()[Input].subset(FixedTest)
print(td.content[0:10].shape)  # 10, 28, 28

inputs = xp.expand_dims(td.content[0:10], -1)

tf = Vit.make(config)
weights = make_weights(tf.params)
pprint(tree_map(lambda _x: _x.shape, weights))
rng = PRNGKey(0)
result = vmap(jit(tf.pipeline), (None, 0, None), 0)(weights, inputs, rng)
