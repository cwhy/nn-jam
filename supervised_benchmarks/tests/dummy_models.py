from typing import NamedTuple, FrozenSet

import numpy as np
from catboost import CatBoostClassifier
from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import FixedTrain, DataUnit, PortSpecs, DataConfig
from supervised_benchmarks.performer_protocol import Performer
from supervised_benchmarks.ports import Port
from supervised_benchmarks.uci_income.consts import AnyNetDiscrete, AnyNetDiscreteOut, AnyNetContinuous


class AnyNetBoostModelConfig(NamedTuple):
    train_data_config: DataConfig

    @staticmethod
    def get_ports() -> PortSpecs:
        return [AnyNetDiscrete, AnyNetContinuous, AnyNetDiscreteOut]

    def prepare(self, repertoire: FrozenSet[Port]) -> Performer:
        print(repertoire)
        assert len(repertoire) == 1, "Only one output port (in a frozenset) is supported"
        out_port = next(iter(repertoire))
        data_pool = self.train_data_config.get_data(self.get_ports())
        tr = data_pool.fixed_subsets[FixedTrain]
        clf = CatBoostClassifier()
        print(tr.content_map[out_port])
        clf.fit(tr.content_map[AnyNetDiscrete], tr.content_map[out_port])

        return BoostPerformer(classifier=clf, repertoire=repertoire)


class BoostPerformer(NamedTuple):
    classifier: CatBoostClassifier
    repertoire: FrozenSet[Port]
    input_port: Port = AnyNetDiscrete

    def perform(self, data_src: DataUnit, tgt: Port) -> NDArray:
        print(data_src[AnyNetDiscrete])
        arr = self.classifier.predict(data_src[self.input_port])
        return np.array(arr)

    def perform_batch(self,
                      data_src: DataUnit,
                      tgt: FrozenSet[Port]) -> DataUnit: ...
