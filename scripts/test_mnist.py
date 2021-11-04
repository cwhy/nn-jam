from pathlib import Path
from typing import List, Dict

import numpy as np
from bokeh.io import show
from bokeh.layouts import row, column
from numpy.typing import NDArray
from variable_protocols.protocols import fmt

from supervised_benchmarks.dataset_protocols import Input, Output, DataPool
from supervised_benchmarks.mnist.mnist import MnistDataConfig, Mnist, \
    FixedTrain, FixedTest, mnist_in_raw, mnist_out_raw
from supervised_benchmarks.mnist.mnist_variations import transformations, MnistConfigIn
from supervised_benchmarks.visualize_utils import view_2d_mono

i = MnistDataConfig(base_path=Path('/Data/torchvision/'))

k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))


mnist_in_flattened = MnistConfigIn(is_float=True, is_flat=True).get_var()
# Test transformations
_ = [print(f'{fmt(s)}, {fmt(t)}') for s, t in transformations.keys()]
# _ = [print(k) for k in transformations.keys()]
tr1 = transformations[(mnist_in_raw, mnist_in_flattened)]
tr2 = transformations[(mnist_in_flattened, mnist_in_raw)]
print(k.array_dict['all.images'].shape)
print(tr1(k.array_dict['all.images']).shape)
print(k.array_dict['all.images'].dtype)
print(tr2(tr1(k.array_dict['all.images'])).dtype)
assert np.all(k.array_dict['all.images'] == tr2(tr1(k.array_dict['all.images'])))

pool_dict = k.retrieve({Input: mnist_in_flattened})
# print(k.data)
pool = pool_dict[Input]
tri = pool.subset(FixedTrain)
print(tri)

pool_dict = k.retrieve({Input: mnist_in_raw, Output: mnist_out_raw})
# print(k.data)
# noinspection PyTypeChecker
# because pycharm sucks
pool_input: DataPool[NDArray] = pool_dict[Input]
# noinspection PyTypeChecker
# because pycharm sucks
pool_output: DataPool[NDArray] = pool_dict[Output]
tr_ft = pool_input.subset(FixedTest)
tr_lb = pool_output.subset(FixedTest)

n_labels = 10
all_labels: Dict[int, int] = dict()
samples: List[int] = []
while len(all_labels) < n_labels:
    s: int = np.random.randint(tr_ft.content.shape[0])
    c = tr_lb.content[s]
    if tr_lb.content[s] not in all_labels:
        all_labels[c] = s
        samples.append(s)
        print(s, c)

figs = []
for lb in sorted(all_labels.keys()):
    s = all_labels[lb]
    ft = tr_ft.content[s, :, :]
    f = view_2d_mono(ft, str(lb))
    figs.append(f)
show(column(row(figs[:5]), row(figs[5:]), sizing_mode="scale_width"))
