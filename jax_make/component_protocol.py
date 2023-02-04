from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Mapping, FrozenSet, Set, NamedTuple, Dict, \
    Tuple, Protocol

from jax import random

import jax_make.params as p
from jax_make.params import RNGKey, ArrayTreeMapping, ArrayParamMapping
from jax import Array as NDArray

Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'

# CompVar = TypeVar("CompVar", bound=str)
X: Literal['X'] = 'X'


class FixedProcess(Protocol):
    @abstractmethod
    def __call__(self, weights: ArrayTreeMapping,
                 inputs: ArrayTreeMapping) -> ArrayTreeMapping: ...


class Process(Protocol):
    def __call__(self, weights: ArrayTreeMapping,
                 inputs: ArrayTreeMapping, rng: Optional[RNGKey]) -> ArrayTreeMapping: ...

class RandomProcess(Protocol):
    def __call__(self, weights: ArrayTreeMapping,
                 inputs: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping: ...

class Pipeline(Protocol):
    def __call__(self, weights: ArrayTreeMapping,
                 x: NDArray, rng: RNGKey) -> NDArray: ...


class FixedPipeline(Protocol):
    def __call__(self, weights: ArrayTreeMapping,
                 x: NDArray) -> NDArray: ...


class NoPipelineFound(Exception):
    ...


class NonFixedComponent(Exception):
    ...


class ProcessPorts(NamedTuple):
    inputs: FrozenSet[str]
    outputs: FrozenSet[str]


def make_ports(inputs: str | Tuple[str, ...], outputs: str | Tuple[str, ...]) -> ProcessPorts:
    if isinstance(inputs, str):
        inputs = (inputs,)
    if isinstance(outputs, str):
        outputs = (outputs,)
    return ProcessPorts(inputs=frozenset(inputs),
                        outputs=frozenset(outputs))


pipeline_ports: ProcessPorts = make_ports(Input, Output)

def random_process2process(process: RandomProcess) -> Process:
    def _fn(weights: ArrayTreeMapping,
            inputs: ArrayTreeMapping, rng: Optional[RNGKey]) -> ArrayTreeMapping:
        if rng is None:
            raise Exception("Trying to run a random process without rng")
        else:
            return process(weights, inputs, rng)

    return _fn

def pipeline2processes(pipeline: Pipeline) -> Dict[ProcessPorts, Process]:
    def _fn(weights: ArrayTreeMapping,
            inputs: ArrayTreeMapping, rng: Optional[RNGKey]) -> ArrayTreeMapping:
        try:
            input_array = p.get_arr(inputs, Input)
        except AssertionError as e:
            raise Exception(f"Failed in converted process from pipeline:", e)
        if rng is None:
            raise Exception("Trying to run pipeline without rng")
        else:
            mp: ArrayTreeMapping = {Output: pipeline(weights, input_array, rng)}
            return mp

    return {pipeline_ports: _fn}


def fixed_pipeline2processes(pipeline: FixedPipeline) -> Dict[ProcessPorts, Process]:
    def _fn(weights: ArrayTreeMapping,
            inputs: ArrayTreeMapping, rng: Optional[RNGKey]) -> ArrayTreeMapping:
        return {Output: pipeline(weights, p.get_arr(inputs, Input))}

    return {pipeline_ports: _fn}


# Fixed_pipeline -> pipeline -> process

# TODO: Component based io generics InputVars, OutputVars
# TODO: Pipeline info: check shape for pipelines
# TODO: check_params as a function of Component to check parameters recursively
@dataclass
class Component:
    weight_params: ArrayParamMapping
    processes: Dict[ProcessPorts, Process]
    is_fixed: bool = False

    def assert_fixed_(self):
        if not self.is_fixed:
            raise NonFixedComponent("The component need to be fixed(non-random) to retrieve fixed process or pipelines")

    def get_pipeline_process(self) -> Process:
        if pipeline_ports in self.processes.keys():
            return self.processes[pipeline_ports]
        else:
            raise NoPipelineFound("There should be a process that is a pipeline (has Input and Output ports)")

    @property
    def pipeline(self) -> Pipeline:
        process = self.get_pipeline_process()

        def _fn(weights: ArrayTreeMapping,
                x: NDArray, rng: RNGKey) -> NDArray:
            return p.get_arr(process(weights, {Input: x}, rng), Output)

        return _fn

    @property
    def fixed_pipeline(self) -> FixedPipeline:
        process = self.get_pipeline_process()
        self.assert_fixed_()

        def _fn(weights: ArrayTreeMapping,
                x: NDArray) -> NDArray:
            return p.get_arr(process(weights, {Input: x}, None), Output)

        return _fn

    def get_fixed_process(self, process_ports: ProcessPorts) -> FixedProcess:
        self.assert_fixed_()

        def _fn(weights: ArrayTreeMapping,
                inputs: ArrayTreeMapping) -> ArrayTreeMapping:
            process = self.processes[process_ports]
            return process(weights, inputs, None)

        return _fn

    @classmethod
    def from_pipeline(cls,
                      params: ArrayParamMapping,
                      pipeline: Pipeline) -> Component:
        return cls(params, pipeline2processes(pipeline))

    @classmethod
    def from_fixed_process(cls,
                           ports_in: Set[str],
                           ports_out: Set[str],
                           params: ArrayParamMapping,
                           process: FixedProcess) -> Component:
        def _fn(weights: ArrayTreeMapping,
                inputs: ArrayTreeMapping, rng: Optional[RNGKey]) -> ArrayTreeMapping:
            return process(weights, inputs)

        return cls(params, {ProcessPorts(frozenset(ports_in), frozenset(ports_out)): _fn}, is_fixed=True)

    @classmethod
    def from_fixed_pipeline(cls,
                            params: ArrayParamMapping,
                            pipeline: FixedPipeline) -> Component:

        def _fn(weights: ArrayTreeMapping,
                inputs: ArrayTreeMapping, rng: Optional[RNGKey]) -> ArrayTreeMapping:
            return {Output: pipeline(weights, p.get_arr(inputs, Input))}

        return cls(params, {pipeline_ports: _fn}, is_fixed=True)


def merge_params(
        components: Mapping[str, Component]
) -> ArrayParamMapping:
    return {k: v.weight_params for k, v in components.items()}


def sequential(components: Mapping[str, Component],
               sequence: List[str]) -> Pipeline:
    pipelines: Dict[str, Pipeline] = {}
    for _comp_name in sequence:
        try:
            pipelines[_comp_name] = components[_comp_name].pipeline
        except NoPipelineFound as e:
            raise Exception(("Sequencing requires component to have pipelines "
                             f"Failed on getting pipeline from component {_comp_name}:"), e)

    def _fn(weights: ArrayTreeMapping,
            x: NDArray, rng: RNGKey) -> NDArray:
        rng, *keys = random.split(rng, len(sequence))
        for comp_name, key in zip(sequence, keys):
            x = pipelines[comp_name](p.get_mapping(weights, comp_name), x, key)
        return x

    return _fn
