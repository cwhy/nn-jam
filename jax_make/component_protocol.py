from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypeVar, Literal, Optional, Generic, Callable, Mapping, FrozenSet, Set, NamedTuple, Dict, Tuple

from jax import random
from numpy.typing import NDArray

from jax_make.params import ArrayTree, RNGKey, ArrayParamTree, ArrayTreeMapping

Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'

CompVar = TypeVar("CompVar", bound=str)
X: Literal['X'] = 'X'


class FixedProcess(Generic[CompVar]):
    def __call__(self, weights: Mapping[CompVar, ArrayTree],
                 inputs: ArrayTreeMapping) -> ArrayTreeMapping: ...


class Process(Generic[CompVar]):
    def __call__(self, weights: Mapping[CompVar, ArrayTree],
                 inputs: ArrayTreeMapping, rng: Optional[RNGKey]) -> ArrayTreeMapping: ...


class Pipeline(Generic[CompVar]):
    def __call__(self, weights: Mapping[CompVar, ArrayTree],
                 x: NDArray, rng: RNGKey) -> NDArray: ...


class FixedPipeline(Generic[CompVar]):
    def __call__(self, weights: Mapping[CompVar, ArrayTree],
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


pipeline_ports = make_ports(Input, Output)


def pipeline2processes(pipeline: Pipeline[CompVar]) -> Dict[ProcessPorts, Process[CompVar]]:
    def _fn(weights: Mapping[CompVar, ArrayTree],
            x: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
        return {Output: pipeline(weights, x[Input], rng)}

    # noinspection PyTypeChecker
    # Because pycharm sucks
    return {pipeline_ports: _fn}


def fixed_pipeline2processes(pipeline: FixedPipeline[CompVar]) -> Dict[ProcessPorts, Process[CompVar]]:
    def _fn(weights: Mapping[CompVar, ArrayTree],
            x: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
        return {Output: pipeline(weights, x[Input])}

    # noinspection PyTypeChecker
    # Because pycharm sucks
    return {pipeline_ports: _fn}


# Fixed_pipeline -> pipeline -> process

# TODO: Component based io generics InputVars, OutputVars
# TODO: Pipeline info: check shape for pipelines
@dataclass
class Component(Generic[CompVar]):
    weight_params: Mapping[CompVar, ArrayParamTree]
    processes: Dict[ProcessPorts, Process[CompVar]]
    is_fixed: bool = False

    def assert_fixed_(self):
        if not self.is_fixed:
            raise NonFixedComponent("The component need to be fixed(non-random) to retrieve fixed process or pipelines")

    def get_pipeline_process(self) -> Process[CompVar]:
        if pipeline_ports in self.processes.keys():
            return self.processes[pipeline_ports]
        else:
            raise NoPipelineFound("There should be a process that is a pipeline (has Input and Output ports)")

    @property
    def pipeline(self) -> Pipeline[CompVar]:
        process = self.get_pipeline_process()

        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: NDArray, key: RNGKey) -> NDArray:
            return process(weights, {Input: x}, key)[Output]

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return _fn

    @property
    def fixed_pipeline(self) -> FixedPipeline[CompVar]:
        process = self.get_pipeline_process()
        self.assert_fixed_()

        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: NDArray) -> NDArray:
            return process(weights, {Input: x}, None)[Output]

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return _fn

    def get_fixed_process(self, process_ports: ProcessPorts[CompVar]) -> Process[CompVar]:
        self.assert_fixed_()

        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: ArrayTreeMapping) -> ArrayTreeMapping:
            process = self.processes[process_ports]
            return process(weights, x, None)

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return _fn

    @classmethod
    def from_pipeline(cls,
                      params: Mapping[CompVar, ArrayParamTree],
                      pipeline: Pipeline[CompVar]) -> Component[CompVar]:
        return cls(params, pipeline2processes(pipeline))

    @classmethod
    def from_fixed_process(cls,
                           ports_in: Set[str],
                           ports_out: Set[str],
                           params: Mapping[CompVar, ArrayParamTree],
                           process: FixedProcess[CompVar]) -> Component[CompVar]:
        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            return process(weights, x)

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return cls(params, {ProcessPorts(frozenset(ports_in), frozenset(ports_out)): _fn}, is_fixed=True)

    @classmethod
    def from_fixed_pipeline(cls,
                            params: Mapping[CompVar, ArrayParamTree],
                            pipeline: FixedPipeline[CompVar]) -> Component[CompVar]:

        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            return {Output: pipeline(weights, x[Input])}

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return cls(params, {pipeline_ports: _fn}, is_fixed=True)


def merge_params(
        components: Mapping[CompVar, Component[CompVar]]
) -> Mapping[CompVar, ArrayParamTree]:
    return {k: v.weight_params for k, v in components.items()}


def sequential(components: Mapping[CompVar, Component[CompVar]],
               sequence: List[CompVar]) -> Pipeline[CompVar]:
    pipelines: Dict[ProcessPorts, Pipeline[CompVar]] = {}
    for _comp_name in sequence:
        try:
            pipelines[_comp_name] = components[_comp_name].pipeline
        except NoPipelineFound as e:
            raise Exception(("Sequencing requires component to have pipelines "
                             f"Failed on getting pipeline from component {_comp_name}:"), e)

    def _fn(weights: Mapping[CompVar, ArrayTree],
            flow_: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
        rng, *keys = random.split(rng, len(sequence))
        for comp_name, key in zip(sequence, keys):
            # noinspection PyTypeChecker
            # Because pycharm sucks
            flow_ = pipelines[comp_name](weights[comp_name], flow_, key)
        return flow_

    # noinspection PyTypeChecker
    # Because pycharm sucks
    return _fn
