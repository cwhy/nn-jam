import jax

from jax_make.params import ArrayTreeMapping, get_arr

Arr = jax.Array


# w:ab, b:b, a -> b
def linear(weights: ArrayTreeMapping, x: Arr) -> Arr:
    return x @ get_arr(weights, 'w') + get_arr(weights, 'b')

