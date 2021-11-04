from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy.typing as npt
from matplotlib import pyplot as plt
from variable_protocols.variables import var_scalar, one_hot

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Input, Output
from supervised_benchmarks.iraven.iraven import get_iraven_, IravenDataConfig, iraven_in_raw, iraven_out_raw, FixedTest
from supervised_benchmarks.metrics import get_mean_acc, get_pair_metric
from supervised_benchmarks.sampler import FullBatchSampler


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

data_config_ = IravenDataConfig(
    task="center_single",
    version="1.0.0",
    size=10000,
    base_path=Path('/media/owner/data/raven'),
    port_vars={
        Input: iraven_in_raw,
        Output: iraven_out_raw,
    })

# noinspection PyTypeChecker
# Because Pycharm sucks
benchmark_config_ = BenchmarkConfig(
    metrics={Output: get_pair_metric('mean_acc', var_scalar(one_hot(10)))},
    on=FixedTest)
benchmark = benchmark_config_.prepare(data_config_.get_data())
sampler = benchmark.sampler
if sampler.tag == 'FullBatchSampler':
    assert isinstance(sampler, FullBatchSampler)
    # noinspection PyTypeChecker
    # Because Pycharm sucks
    sample_input = sampler.full_batch[Input][0, :, :, :]
    show_sample_(sample_input, "tst")
