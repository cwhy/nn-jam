import jax.numpy as xp
from jax import jit, make_jaxpr

from tests.einops_utils import mix, MixWeights, init_weights
from tests.jax_modules.multi_head_attn import MultiHeadAttn

b, c, x, x1 = 4, 5, 2, 3
k = mix("b x c -> b x1 c", weight_shape='x x1', bias_shape='x1', x=x, x1=x1)
print(k)
print("finished jit")
weights = MixWeights(w=xp.ones(k.weight_shape), b=xp.ones(k.bias_shape))
k = k.make()[1](weights, xp.ones((b, x, c)))
print(k.shape)

H, T, K, C, N = 4, 5, 8, 10, 3
inputs = xp.ones((C, N, T))


mha_config = MultiHeadAttn(
    n_seq=T,
    n_data=N,
    n_heads=H,
    dim_model=K,
    dim_input=C)
mha = MultiHeadAttn.make(mha_config)
weights = init_weights(mha.weight_params)
out_expr = make_jaxpr(jit(mha.process))(weights, inputs)
out = mha.process(weights, inputs)
print(out_expr)
print(out.shape)
print(mha.weight_params)
