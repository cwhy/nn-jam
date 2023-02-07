import jax

from jax_make.component_protocol import Component
from jax_make.params import ArrayTreeMapping, get_arr, WeightParams

Arr = jax.Array


# Pipeline: w:ab, b:b, a -> b
def linear(weights: ArrayTreeMapping, x: Arr) -> Arr:
    return x @ get_arr(weights, 'w') + get_arr(weights, 'b')


def linear_component(n_in: int, n_out: int) -> Component:
    return Component.from_fixed_pipeline(
        {"w": WeightParams(shape=(n_in, n_out)),
         "b": WeightParams(shape=(n_out,), init=0)},
        linear
    )
