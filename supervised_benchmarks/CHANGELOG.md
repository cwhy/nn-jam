# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2022-05-02
- [Subset] Removed index requirement for Subset
- [Subset] Made FixedSubset dataset specifics, leave 'FixedSubsetType' shared
- [Port] Refactored port, and used generic Port, Input/Output are no longer strings, but port types
- Use Python 3.8 style type annotation by default from now on
- Ditch Pycharm typing hints, use mypy instead from now on