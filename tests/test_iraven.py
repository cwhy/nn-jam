from pathlib import Path
from typing import List, Dict

import numpy as np
from bokeh.io import show
from bokeh.layouts import row, column
from numpy.typing import NDArray
from variable_protocols.protocols import fmt

from supervised_benchmarks.dataset_protocols import Input, Output, DataPool
from supervised_benchmarks.iraven.iraven import get_iraven_
from supervised_benchmarks.mnist.mnist import MnistDataConfig, Mnist, \
    FixedTrain, FixedTest, mnist_in_raw, mnist_out_raw
from supervised_benchmarks.mnist.mnist_variations import transformations, MnistConfigIn
from supervised_benchmarks.visualize_utils import view_2d_mono
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.typing as npt


def show_sample_(img: npt.NDArray, name: str) -> None:
    fig = plt.figure(figsize=(4.5, 6.9))
    heights = (5, 2)
    outer = gridspec.GridSpec(
        2, 1,
        wspace=0.2,
        hspace=0.2,
        height_ratios=heights)

    shapes = [(3, 3), (2, 4)]
    for j in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(
            *shapes[j],
            subplot_spec=outer[j],
            wspace=0.1, hspace=0.1)
        for i in range(9):
            if i != 8:
                ax = plt.Subplot(fig, inner[i // shapes[j][1], i % shapes[j][1]])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(img[i + j * 8, :, :], plt.get_cmap('gist_heat'))
                fig.add_subplot(ax)
    fig.savefig(f"{name}view.pdf")

data_dict = get_iraven_(Path("/media/owner/data/raven"), "1.0.0", 10000, "center_single")
