from pathlib import Path
from typing import NamedTuple, FrozenSet, Literal

import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from numpy.typing import NDArray

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import FixedTrain, FixedTest, DataUnit, PortSpecs
from supervised_benchmarks.metrics import get_pair_metric
from supervised_benchmarks.ports import Port
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import FixedEpochSamplerConfig, FullBatchSamplerConfig
from supervised_benchmarks.tabular_utils import ColumnInfo, NumStats, AnyNetStrategyConfig, parse_polars, \
    anynet_load_polars
from supervised_benchmarks.uci_income.consts import AnyNetDiscrete, AnyNetDiscreteOut, variable_names, AnyNetContinuous
from supervised_benchmarks.uci_income.uci_income import UciIncomeDataConfig, UciIncome
from variable_protocols.tensorhub import F, V


class BoostModelConfig(NamedTuple):
    ports: PortSpecs

    type: Literal['ModelConfig'] = 'ModelConfig'

    def prepare(self) -> Performer:
        query = [AnyNetDiscrete, AnyNetContinuous, AnyNetDiscreteOut]
        config = AnyNetStrategyConfig()
        data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                          column_config=config,
                                          query=query)
        data_pool = data_config.get_data()
        tr = data_pool.fixed_subsets[FixedTrain]
        clf = CatBoostClassifier()
        print(tr.content_map[AnyNetDiscreteOut])
        clf.fit(tr.content_map[AnyNetDiscrete], tr.content_map[AnyNetDiscreteOut])

        return BoostPerformer(classifier=clf, repertoire=frozenset({AnyNetDiscreteOut}))


class BoostPerformer(NamedTuple):
    classifier: CatBoostClassifier
    repertoire: FrozenSet[Port]

    def perform(self, data_src: DataUnit, tgt: Port) -> NDArray:
        print(data_src[AnyNetDiscrete])
        arr = self.classifier.predict(data_src[AnyNetDiscrete])
        return np.array(arr)

    def perform_batch(self,
                      data_src: DataUnit,
                      tgt: FrozenSet[Port]) -> DataUnit: ...


def test_polars():
    data = pl.read_csv('/Data/uci/adult.data', delimiter=',', has_header=False, new_columns=variable_names)
    config = AnyNetStrategyConfig()
    variable_protocols = parse_polars(config, data)
    print(variable_protocols.fmt())
    res = anynet_load_polars(config, data)
    print(res)


def test_utils():
    base_path = Path('/Data/uci')
    config = AnyNetStrategyConfig()
    data_class = UciIncome(base_path, config)
    discrete_labels = [l for i, l in enumerate(variable_names) if data_class.data_info.is_digits[i]]
    continuous_labels = [l for i, l in enumerate(variable_names) if not data_class.data_info.is_digits[i]]
    print(data_class.data_info.is_digits)
    print(discrete_labels)
    print(continuous_labels)


def test_data_init():
    base_path = Path('/Data/uci')
    config = AnyNetStrategyConfig()
    data_class = UciIncome(base_path, config)
    print(config)


def test_data_fixed():
    query = [AnyNetDiscrete, AnyNetContinuous, AnyNetDiscreteOut]
    config = AnyNetStrategyConfig()
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                      column_config=config,
                                      query=query)
    data_pool = data_config.get_data()
    tr = data_pool.fixed_subsets[FixedTrain]
    tst = data_pool.fixed_subsets[FixedTest]
    print(tr.content_map)
    print(tst.content_map)
    assert AnyNetDiscrete in tr.content_map
    assert AnyNetDiscrete in tst.content_map
    assert list(tr.content_map.keys()) == list(tst.content_map.keys()) == query
    print(tr.content_map[AnyNetDiscrete].shape)
    print(tst.content_map[AnyNetDiscrete].shape)
    assert tr.content_map[AnyNetDiscrete].shape == (32561, 13)
    assert tst.content_map[AnyNetDiscrete].shape == (16281, 13)
    print(tr.content_map[AnyNetDiscreteOut].shape)
    assert tr.content_map[AnyNetDiscreteOut].shape == (32561, 1)
    print(tr.content_map[AnyNetContinuous].shape)
    assert tr.content_map[AnyNetContinuous].shape == (32561, 1)
    sampler_config = FixedEpochSamplerConfig(512)
    sampler = sampler_config.get_sampler(tr)
    mini_batch = next(sampler.iter)
    assert AnyNetDiscrete in mini_batch
    assert mini_batch[AnyNetDiscrete].shape == (512, 13)
    test_sampler = FullBatchSamplerConfig().get_sampler(tst)
    full_test = test_sampler.full_batch
    assert AnyNetDiscrete in full_test
    assert full_test[AnyNetDiscrete].shape == (16281, 13)
    print(mini_batch[AnyNetContinuous].shape)
    assert mini_batch[AnyNetContinuous].shape == (512, 1)
    assert full_test[AnyNetContinuous].shape == (16281, 1)

    # bench = BenchmarkConfig(metrics={AnyNetDiscreteOut: get_pair_metric('mean_acc', var_scalar(ordinal(n_tokens)))},
    #                         on=FixedTrain).bench(data_config)


def test_boost_init():
    query = [AnyNetDiscrete, AnyNetContinuous, AnyNetDiscreteOut]
    config = AnyNetStrategyConfig()
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                      column_config=config,
                                      query=query)
    data_pool = data_config.get_data()
    tst = data_pool.fixed_subsets[FixedTest]

    sampler_config = FixedEpochSamplerConfig(512)
    sampler = sampler_config.get_sampler(tst)

    mini_batch = next(sampler.iter)
    config = BoostModelConfig(ports=query)
    performer = config.prepare()
    result = performer.perform(data_src=mini_batch, tgt=AnyNetDiscreteOut)
    print((result == mini_batch[AnyNetDiscreteOut]).mean())


def test_benchmark():
    query = [AnyNetDiscrete, AnyNetContinuous, AnyNetDiscreteOut]
    config = AnyNetStrategyConfig()
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                      column_config=config,
                                      query=query)
    benchmark_config = BenchmarkConfig(
        metrics={AnyNetDiscreteOut: get_pair_metric('mean_acc', AnyNetDiscreteOut.protocol)},
        on=FixedTest)
    model_config = BoostModelConfig(
        ports=[AnyNetDiscreteOut, AnyNetDiscrete])
    performer = model_config.prepare()
    z = benchmark_config.bench(data_config, performer)
    print(z)
