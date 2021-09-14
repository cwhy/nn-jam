import numpy as np
from numpy.typing import NDArray
from variable_protocols.variables import var_scalar, ordinal

from supervised_benchmarks.metric_protocols import MetricResult, PairMetricImp


# noinspection PyTypeChecker
# Because Pycharm sucks
def get_mean_acc(n_category: int):
    # noinspection PyTypeChecker
    # Because Pycharm sucks
    def mean_acc_numpy(output: NDArray, target: NDArray):
        target_class = np.argmax(target, axis=1)
        output_class = np.argmax(output, axis=1)
        return MetricResult(
            content=np.mean(output_class == target_class).item(),
            result_type='mean_acc')

    return PairMetricImp(
        protocol=var_scalar(ordinal(n_category)),
        type='mean_acc',
        measure=mean_acc_numpy
    )
