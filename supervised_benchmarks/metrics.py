import numpy as np
from numpy.typing import NDArray

from supervised_benchmarks.metric_protocols import MetricResult, PairMetricImp, PairMetricType
from variable_protocols.tensorhub import Tensor


def get_pair_metric(metric_type: PairMetricType, protocol: Tensor):
    if metric_type == "mean_acc":
        # TODO: better assertion
        assert isinstance(protocol, Tensor)
        if protocol.base.type_name == 'ordinal' or protocol.base.type_name == 'ids':
            def mean_acc(output: NDArray, target: NDArray) -> MetricResult:
                return MetricResult(
                    content=np.mean(output == target).item(),
                    result_type=metric_type)
        elif protocol.base.type_name == '1hot':
            def mean_acc(output: NDArray, target: NDArray) -> MetricResult:
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
            measure=mean_acc
        )
    else:
        raise NotImplementedError(f"metric type {metric_type} is not implemented yet")
