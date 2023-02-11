
import jax

from jax_make.component_protocol import Component
from jax_make.params import ArrayTreeMapping, get_arr, WeightParams

Arr = jax.Array
def rwkv(weights: ArrayTreeMapping, x: Arr) -> Arr:
    r, w, k, v = get_arr(weights, 'r'), get_arr(weights, 'w'), get_arr(weights, 'k'), get_arr(weights, 'v')
    pass



def rwkv_component(n_in: int, n_out: int) -> Component:
    pass
