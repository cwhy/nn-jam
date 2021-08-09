from pathlib import Path

import numpy as np
from bokeh.io import show
from einops import rearrange, repeat

from supervised_benchmarks.dataset_protocols import Input, FixedTrain
from supervised_benchmarks.mnist import MnistDataConfig, Mnist, mnist_in, mnist_in_flattened, transformations
from supervised_benchmarks.visualize_utils import view_2d_mono

i = MnistDataConfig(base_path=Path('/Data/torchvision/'))

k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))


pool_dict = k.retrieve({Input: mnist_in})
# print(k.data)
pool = pool_dict[Input]
tri = pool.subset(FixedTrain)
print(tri)
p = view_2d_mono(np.hstack([tri.content[109, :, :], tri.content[129, :, :]]))
show(p)
p = view_2d_mono(np.vstack([tri.content[109, :, :], tri.content[129, :, :]]))
show(p)

