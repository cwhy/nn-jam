# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## Long Term TODO
- Variable protocol DSL
- Enable Generics in Variable Protocol

## Mid Term TODO : 
- Fix mnist
- Fix iraven
- add modeltesting utility

## TODO : Get UCI working
- new ver
  
## [0.0.5] - 2022-07-15
- [UCI_INCOME] 
  - Test UCI with benchmark and metrics
- [Benchmark]
  - Now using `FixSubsetType` Instead of `Subset` for subset options

## [0.0.4] - 2022-07-13
- [UCI_INCOME] 
  - Fixed tests
  - added tests using catboost

## [0.0.3] - 2022-05-06
- [Model]: Added sample model

## [0.0.2] - 2022-05-06
- [DataSubset] Renamed to [DataSubset]
- [DataSubset] .content becomes .content_map and returns a map of port:array
- [DataSubset] removed `.variable` since variable is now tied to Ports
- [DataPool] now hosts only queried data, removed `src_var` and `tgt_var`
- [DataContent] removed, numpy becomes first citizen
- [Sampler] MiniBatchSampler/FixedEpochSampler returns iterator of map of port:array(DataUnit) now
            instead of map of port:iterator
- [Port] Fixed wrong implementation of ports
- [DataSet] Changed `.port` to `.export`
- [DataConfig] Renamed `.port_vars` to `.query`
- Discard Variable Protocol details first

## [0.0.1] - 2022-05-02
- [Subset] Removed index requirement for Subset
- [Subset/DataPool] Made FixedSubset dataset specifics, leave 'FixedSubsetType' shared
- [Port] Refactored port, and used generic Port, Input/Output are no longer strings, but port types
- Use Python 3.8 style type annotation by default from now on
- Ditch Pycharm typing hints, use mypy instead from now on
  because pyCharm sucks https://youtrack.jetbrains.com/issue/PY-49439
- Not runnable
- Before refactoring
    - [Data] content is a map instead of DataContent
    - [DataContents] remove all the variants of generics, abstract it out
    - [DataContents] now a wrapper container of arrays
