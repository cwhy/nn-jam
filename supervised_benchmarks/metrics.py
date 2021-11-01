import numpy as np
from numpy.typing import NDArray
from variable_protocols.base_variables import BaseVariable
from variable_protocols.protocols import VariableTensor, Variable
from variable_protocols.variables import var_scalar, ordinal, one_hot

from supervised_benchmarks.metric_protocols import MetricResult, PairMetricImp, PairMetricType


# noinspection PyTypeChecker
# Because Pycharm sucks
def get_pair_metric(metric_type: PairMetricType, protocol: Variable):
    if metric_type == "mean_acc":
        assert isinstance(protocol, VariableTensor)
        assert isinstance(protocol.var, BaseVariable)
        if protocol.var.type_name == 'ordinal':
            def mean_acc_numpy(output_class: NDArray, target_class: NDArray):
                return MetricResult(
                    content=np.mean(output_class == target_class).item(),
                    result_type=metric_type)
        elif protocol.var.type_name == '1hot':
            def mean_acc_numpy(output: NDArray, target: NDArray):
                target_class = np.argmax(target, axis=1)
                output_class = np.argmax(output, axis=1)
                return MetricResult(
                    content=np.mean(output_class == target_class).item(),
                    result_type='mean_acc')
        else:
            raise Exception(f"protocol {protocol} is not supported for metric type {metric_type}")
        return PairMetricImp(
            protocol=protocol,
            type=metric_type,
            measure=mean_acc_numpy
        )
    else:
        raise NotImplementedError(f"metric type {metric_type} is not implemented yet")

#
#     return PairMetricImp(
#         protocol=var_scalar(one_hot(n_category)),
#         type='mean_acc',
#         measure=mean_acc_numpy
#     )


# def accuracy(output: NDArray, target: NDArray) -> float:
#     target_class = jnp.argmax(target, axis=1)
#     output_class = jnp.argmax(output, axis=1)
#     return jnp.mean(output_class == target_class)
