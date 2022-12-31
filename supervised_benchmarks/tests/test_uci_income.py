from pathlib import Path

import polars as pl

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import FixedTrain, FixedTest
from supervised_benchmarks.metrics import get_pair_metric
from supervised_benchmarks.sampler import FixedEpochSamplerConfig, FullBatchSamplerConfig
from supervised_benchmarks.tabular_utils import AnyNetStrategyConfig, parse_polars, \
    anynet_load_polars
from supervised_benchmarks.tests.dummy_models import AnyNetBoostModelConfig
from supervised_benchmarks.uci_income.consts import AnyNetDiscrete, AnyNetDiscreteOut, variable_names, AnyNetContinuous
from supervised_benchmarks.uci_income.uci_income import UciIncomeDataConfig, UciIncome


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
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                      column_config=AnyNetStrategyConfig())
    model_config = AnyNetBoostModelConfig(train_data_config=data_config)
    data_pool = data_config.get_data(model_config.get_ports())
    tst = data_pool.fixed_subsets[FixedTest]

    sampler_config = FixedEpochSamplerConfig(512)
    sampler = sampler_config.get_sampler(tst)

    mini_batch = next(sampler.iter)
    performer = model_config.prepare(AnyNetDiscreteOut)
    result = performer.perform(data_src=mini_batch, tgt=AnyNetDiscreteOut)
    print((result == mini_batch[AnyNetDiscreteOut]).mean())


def test_benchmark():
    config = AnyNetStrategyConfig()
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                      column_config=config)

    model_config = AnyNetBoostModelConfig(train_data_config=data_config)
    benchmark_config = BenchmarkConfig(
        on_metrics={AnyNetDiscreteOut: get_pair_metric('mean_acc', AnyNetDiscreteOut.protocol)},
        on_data=FixedTest,
        model_config=model_config,
        data_config=data_config)
    z = benchmark_config.bench()
    print(z)
