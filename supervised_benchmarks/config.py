from variables import VariableGroup, VariableTensor
from typing import Callable, NamedTuple, Literal, Generic, TypeVar

FeatureLayouts = Literal["flatten"]
Input = TypeVar('Input')
Output = TypeVar('Output')
Results = TypeVar('Results')


class Data():
    pass


mnist_in = VariableGroup(name="mnist_in",
                         variables={
                             VariableTensor(Bounded(max=1, min=0), (28, 28))
                         })
mnist_out = VariableGroup(name="mnist_out",
                          variables={
                              VariableTensor(OneHot(n_category=10), (1,))
                          })


class BenchEnv(NamedTuple, Generic[Input, Output]):
    class Model(NamedTuple):
        pass

    model: Model
    train: Callable[[Model, Data], Model]
    test: Callable[[Model], Results]
    name: str
