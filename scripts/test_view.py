from pathlib import Path

import numpy as np
from bokeh.io import show

from supervised_benchmarks.ports import Input
from supervised_benchmarks.mnist.mnist import MnistDataConfig, Mnist, mnist_in_raw, FixedTrain
from supervised_benchmarks.visualize_utils import view_2d_mono

i = MnistDataConfig(base_path=Path('/Data/torchvision/'))

k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))


pool_dict = k.retrieve({Input: mnist_in_raw})
# print(k.data)
pool = pool_dict[Input]
tri = pool.subset(FixedTrain)
print(tri)
p = view_2d_mono(np.hstack([tri.content[109, :, :], tri.content[129, :, :]]))
show(p)
p = view_2d_mono(np.vstack([tri.content[109, :, :], tri.content[129, :, :]]))
show(p)

