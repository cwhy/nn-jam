from pathlib import Path
from typing import Set, List, Dict

import numpy as np
from bokeh.io import show
from bokeh.layouts import row, column
from einops import rearrange, repeat

from supervised_benchmarks.dataset_protocols import Input, FixedTrain, Output
from supervised_benchmarks.mnist import MnistDataConfig, Mnist, mnist_in, mnist_in_flattened, transformations, mnist_out
from supervised_benchmarks.visualize_utils import view_2d_mono

i = MnistDataConfig(base_path=Path('/Data/torchvision/'))

k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))

# Test transformation
tr1 = transformations[(mnist_in, mnist_in_flattened)]
tr2 = transformations[(mnist_in_flattened, mnist_in)]
print(k.array_dict['train.images'].shape)
print(tr1(k.array_dict['train.images']).shape)
assert np.all(k.array_dict['train.images'] == tr2(tr1(k.array_dict['train.images'])))

pool_dict = k.retrieve({Input: mnist_in_flattened})
# print(k.data)
pool = pool_dict[Input]
tri = pool.subset(FixedTrain)
print(tri)

pool_dict = k.retrieve({Input: mnist_in, Output: mnist_out})
# print(k.data)
pool_input = pool_dict[Input]
tr_ft = pool_input.subset(FixedTrain)
pool_output = pool_dict[Output]
tr_lb = pool_output.subset(FixedTrain)

n_labels = 10
all_labels: Dict[int, int] = dict()
samples: List[int] = []
while len(all_labels) < n_labels:
    s = np.random.randint(tr_ft.content.shape[0])
    if tr_lb.content[s] not in all_labels:
        all_labels[tr_lb.content[s]] = s
        samples.append(s)
        print(s, tr_lb.content[s])

figs = []
for lb in sorted(all_labels.keys()):
    s = all_labels[lb]
    ft = tr_ft.content[s, :, :]
    f = view_2d_mono(ft)
    figs.append(f)
show(column(row(figs[:5]), row(figs[5:]), sizing_mode="scale_width"))

