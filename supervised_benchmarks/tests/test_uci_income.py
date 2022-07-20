from pathlib import Path
from typing import NamedTuple, FrozenSet, Mapping, Literal

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
from supervised_benchmarks.tabular_utils import ColumnStats, NumStats, AnyNetStrategyConfig
from supervised_benchmarks.uci_income.consts import AnyNetDiscrete, AnyNetDiscreteOut, variable_names
from supervised_benchmarks.uci_income.uci_income import UciIncomeDataConfig, UciIncome


class BoostModelConfig(NamedTuple):
    ports: PortSpecs

    type: Literal['ModelConfig'] = 'ModelConfig'

    def prepare(self) -> Performer:
        query = {AnyNetDiscrete: uci_income_in_anynet_discrete,
                 AnyNetDiscreteOut: uci_income_out_anynet_discrete}
        data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'), query=query)
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
    cols = data.get_columns()
    config = AnyNetStrategyConfig()
    for col in cols:
        if col.is_utf8():
            num_stats = None
        else:
            num_stats = NumStats(mean=col.mean(), std=col.std(), min=col.min(), max=col.max())
        stats = ColumnStats(
            n=col.len(),
            n_unique=len(col.unique()),
            n_null=col.null_count(),
            sorted_desc_unique_count=col.unique_counts().sort(reverse=True).to_list(),
            num_stats=num_stats
        )
        base = config.classify_column(stats)
        print(base)


def test_utils():
    base_path = Path('/Data/uci')
    data_class = UciIncome(base_path)
    discrete_labels = [l for i, l in enumerate(variable_names) if data_class.data_info.is_digits[i]]
    continuous_labels = [l for i, l in enumerate(variable_names) if not data_class.data_info.is_digits[i]]
    print(data_class.data_info.is_digits)
    print(discrete_labels)
    print(continuous_labels)


def test_get_data():
    query = {AnyNetDiscrete: uci_income_in_anynet_discrete,
             AnyNetDiscreteOut: uci_income_out_anynet_discrete}
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'), query=query)
    data_pool = data_config.get_data()
    tr = data_pool.fixed_subsets[FixedTrain]
    tst = data_pool.fixed_subsets[FixedTest]
    assert AnyNetDiscrete in tr.content_map
    assert AnyNetDiscrete in tst.content_map
    assert tr.query == tst.query == query
    print(tr.content_map[AnyNetDiscrete].shape)
    print(tst.content_map[AnyNetDiscrete].shape)
    assert tr.content_map[AnyNetDiscrete].shape == (32561, 14)
    assert tst.content_map[AnyNetDiscrete].shape == (16281, 14)
    print(tr.content_map[AnyNetDiscreteOut].shape)
    assert tr.content_map[AnyNetDiscreteOut].shape == (32561,)
    sampler_config = FixedEpochSamplerConfig(512)
    sampler = sampler_config.get_sampler(tr)

    mini_batch = next(sampler.iter)
    assert AnyNetDiscrete in mini_batch
    assert mini_batch[AnyNetDiscrete].shape == (512, 14)
    test_sampler = FullBatchSamplerConfig().get_sampler(tst)
    full_test = test_sampler.full_batch
    assert AnyNetDiscrete in full_test
    assert full_test[AnyNetDiscrete].shape == (16281, 14)
    # bench = BenchmarkConfig(metrics={AnyNetDiscreteOut: get_pair_metric('mean_acc', var_scalar(ordinal(n_tokens)))},
    #                         on=FixedTrain).bench(data_config)

    assert True


def test_boost_init():
    query = {AnyNetDiscrete: uci_income_in_anynet_discrete,
             AnyNetDiscreteOut: uci_income_out_anynet_discrete}
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'), query=query)
    data_pool = data_config.get_data()
    tst = data_pool.fixed_subsets[FixedTest]
    sampler_config = FixedEpochSamplerConfig(512)
    sampler = sampler_config.get_sampler(tst)

    mini_batch = next(sampler.iter)
    config = BoostModelConfig(
        ports={AnyNetDiscreteOut: uci_income_out_anynet_discrete, AnyNetDiscrete: uci_income_in_anynet_discrete})
    performer = config.prepare()
    result = performer.perform(data_src=mini_batch, tgt=AnyNetDiscreteOut)
    print((result == mini_batch[AnyNetDiscreteOut]).mean())


def test_benchmark():
    query = {AnyNetDiscrete: uci_income_in_anynet_discrete,
             AnyNetDiscreteOut: uci_income_out_anynet_discrete}
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'), query=query)
    benchmark_config = BenchmarkConfig(
        metrics={AnyNetDiscreteOut: get_pair_metric('mean_acc', data_config.query[AnyNetDiscreteOut])},
        on=FixedTest)
    model_config = BoostModelConfig(
        ports={AnyNetDiscreteOut: uci_income_out_anynet_discrete, AnyNetDiscrete: uci_income_in_anynet_discrete})
    performer = model_config.prepare()
    z = benchmark_config.bench(data_config, performer)
    print(z)
