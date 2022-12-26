from typing import NamedTuple, FrozenSet, Literal

import numpy as np
from catboost import CatBoostClassifier
from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import FixedTrain, DataUnit, PortSpecs, DataConfig
from supervised_benchmarks.ports import Port
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.uci_income.consts import AnyNetDiscrete, AnyNetDiscreteOut


class AnyNetBoostModelConfig(NamedTuple):
    ports: PortSpecs
    train_data_config: DataConfig

    type: Literal['ModelConfig'] = 'ModelConfig'

    def prepare(self) -> Performer:
        data_pool = self.train_data_config.get_data()
        tr = data_pool.fixed_subsets[FixedTrain]
        clf = CatBoostClassifier()
        print(tr.content_map[AnyNetDiscreteOut])
        clf.fit(tr.content_map[AnyNetDiscrete], tr.content_map[AnyNetDiscreteOut])

        return BoostPerformer(classifier=clf, repertoire=frozenset({AnyNetDiscreteOut}), input_port=AnyNetDiscrete)


class BoostPerformer(NamedTuple):
    classifier: CatBoostClassifier
    repertoire: FrozenSet[Port]
    input_port: Port

    def perform(self, data_src: DataUnit, tgt: Port) -> NDArray:
        print(data_src[AnyNetDiscrete])
        arr = self.classifier.predict(data_src[self.input_port])
        return np.array(arr)

    def perform_batch(self,
                      data_src: DataUnit,
                      tgt: FrozenSet[Port]) -> DataUnit: ...
