# Audition(CWhy Supervised Learning Benchmark)
## Intro
Audition (CWhy Supervised Learning Benchmark) is my personal benchmark to measure perfomance of supervised learning.
Risen from ashes of cwhy/MLKit(Rip TF1.0x, why google? 😖)

## Non Goals
* make a unbiased benchmark for everyone
* measure computational resources of algorithms
* no-friction plugin for existing code
* measure bandit/RL
* ImageNet-level large datasets that poor souls can't afford
* super efficient training

## Final Goals
* measure supervised learning with my biased view
* no nn framework/compute library dependency
* very handy to plug-in anything Python by wrappers
* support cwhy/anynet input/output configurations
* nice debug messages thoughout the whole experience
* utilize Python to its finest

## Current Remarks
* quick and dirty version for testing cwhy/anynet
* types are a must, no array type yet
* dataconfig type won't deal with transformations, only query

## Short Term Goals
* iraven on transformer
* cifar, tabular datasets from UCI and stuff
* input output validation

## Acknowledgement
* Download helper functions are from https://github.com/pytorch/vision
