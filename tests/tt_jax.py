import jax.numpy as xp

from tests.einops_utils import mix, MixWeights, init_weights
from tests.jax_modules import MultiHeadAttnBK, MultiHeadAttnWeights, MultiHeadAttn

H, T, K, C, N = 4, 5, 8, 10, 3
mha = MultiHeadAttnBK(
    n_seq=T,
    n_data=N,
    n_heads=H,
    dim_model=K,
    dim_input=C,
    w_k=xp.ones((K, C)),
    w_q=xp.ones((K, C)),
    w_v=xp.ones((K, C)),
    out_linear=xp.ones((K, K))
)

inputs = xp.ones((C, N, T))
outs = mha.forward_self(inputs)
print(outs.shape)

b, c, x, x1 = 4, 5, 2, 3
k = mix("b x c -> b x1 c", weight_shape='x x1', bias_shape='x1', x=x, x1=x1)
print(k)
print("finished jit")
weights = MixWeights(w=xp.ones(k.weight_shape), b=xp.ones(k.bias_shape))
k = k.make()[1](weights, xp.ones((b, x, c)))
print(k.shape)

mha = MultiHeadAttn(
    n_seq=T,
    n_data=N,
    n_heads=H,
    dim_model=K,
    dim_input=C).make()
weights = init_weights(mha.weight_params)
out = mha.process(weights, inputs)
print((out == outs).all())
print(mha.weight_params)
