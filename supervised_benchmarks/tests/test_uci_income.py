from pathlib import Path

from supervised_benchmarks.benchmark import BenchmarkConfig
from supervised_benchmarks.dataset_protocols import FixedTrain, FixedTest
from supervised_benchmarks.metrics import get_pair_metric
from supervised_benchmarks.sampler import FixedEpochSamplerConfig, FullBatchSampler, FullBatchSamplerConfig
from supervised_benchmarks.uci_income.consts import AnyNetDiscrete, AnyNetDiscreteOut, n_tokens
from supervised_benchmarks.uci_income.uci_income import UciIncomeDataConfig, uci_income_in_anynet_discrete, \
    uci_income_out_anynet_discrete
from variable_protocols.variables import var_scalar, ordinal


def test_get_data():
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                      query={AnyNetDiscrete: uci_income_in_anynet_discrete,
                                             AnyNetDiscreteOut: uci_income_out_anynet_discrete})
    data_pool = data_config.get_data()
    tr = data_pool.fixed_subsets[FixedTrain]
    tst = data_pool.fixed_subsets[FixedTest]
    assert AnyNetDiscrete in tr.content_map
    assert AnyNetDiscrete in tst.content_map
    assert tr.query == tst.query == {AnyNetDiscrete: uci_income_in_anynet_discrete}
    print(tr.content_map[AnyNetDiscrete].shape)
    print(tst.content_map[AnyNetDiscrete].shape)
    assert tr.content_map[AnyNetDiscrete].shape == (32560, 15)
    assert tst.content_map[AnyNetDiscrete].shape == (16280, 15)
    sampler_config = FixedEpochSamplerConfig(512)
    sampler = sampler_config.get_sampler(tr)
    mini_batch = next(sampler.iter)
    assert AnyNetDiscrete in mini_batch
    assert mini_batch[AnyNetDiscrete].shape == (512, 15)
    test_sampler = FullBatchSamplerConfig().get_sampler(tst)
    full_test = test_sampler.full_batch
    assert AnyNetDiscrete in full_test
    assert full_test[AnyNetDiscrete].shape == (16280, 15)
    # bench = BenchmarkConfig(metrics={AnyNetDiscreteOut: get_pair_metric('mean_acc', var_scalar(ordinal(n_tokens)))},
    #                         on=FixedTrain).bench(data_config)

    assert True
