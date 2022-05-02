from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import List

import jax
import jax.numpy as xp
import numpy as np
from bokeh.io import show
from bokeh.layouts import column, row
from einops import rearrange
from jax import tree_map, vmap
from jax._src.random import PRNGKey

from supervised_benchmarks.dataset_protocols import Data
from supervised_benchmarks.ports import Input, Output
from supervised_benchmarks.mnist.mnist import MnistDataConfig, Mnist, FixedTest
from supervised_benchmarks.mnist.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.visualize_utils import view_2d_mono, view_img_rgba
from jax_make.components.dirty_patches import DirtyPatches
from jax_make.params import make_weights

data_config_ = MnistDataConfig(
    base_path=Path('/Data/torchvision/'),
    port_vars={
        Input: MnistConfigIn(is_float=True, is_flat=False).get_var(),
        Output: MnistConfigOut(is_1hot=True).get_var()
    })
td: Data = data_config_.get_data()[Input].subset(FixedTest)
print(td.content[0:10].shape)  # 10, 28, 28
r = xp.moveaxis(xp.stack((td.content[0:10], td.content[1:11], td.content[2:12], xp.ones((10, 28, 28)))), 0, -1)
print(r.shape)  # 10, 28, 28, 4

# n h w c
dim_h = dim_w = ch = 4
patches = jax.lax.conv_general_dilated_patches(
    lhs=r,
    filter_shape=(dim_h, dim_w),
    window_strides=(dim_h, dim_w),
    padding='VALID',
    rhs_dilation=(1, 1),
    dimension_numbers=("NHWC", "OIHW", "NHWC")
)

out = rearrange(patches, 'n h w (c ph pw)-> n h w (ph pw c)',
                c=ch, ph=dim_h, pw=dim_w)

show(view_img_rgba(255 * np.asarray(r[0, :, :, :])))
outnp = np.asarray(out[0, :, :, :])
rows = []
for i in range(7):
    column_list = []
    for j in range(7):
        f = view_img_rgba(255 * outnp[j, i, :].reshape(4, 4, 4), f"{i}, {j}")
        column_list.append(f)
    rows.append(column(*column_list))

show(row(*rows))

patches = DirtyPatches.make(DirtyPatches(
    dim_out=12,
    n_sections_w=7,
    n_sections_h=7,
    w=28,
    h=28,
    ch=4,
    mlp_n_hidden=[11],
    mlp_activation="relu",
    dropout_keep_rate=1
))

pprint(patches.params)
weights = make_weights(patches.params)
pprint(tree_map(lambda _x: _x.shape, weights))
rng = PRNGKey(0)
out = vmap(patches.process, (None, 0, None), 0)(weights, r, rng)
print(out.shape)
