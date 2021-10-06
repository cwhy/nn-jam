import jax.numpy as xp

from tests.einops_utils import mix, MixWeights
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
weights = MixWeights(xp.ones(k.weight_shape), xp.ones(k.bias_shape))
k = k.process(weights, xp.ones((b, x, c)))
print(k.shape)

mix_weights = MixWeights(xp.ones((C, K)), xp.ones(K))
out_weights = MixWeights(xp.ones((K, K)), xp.ones(K))
weights = MultiHeadAttnWeights(mix_weights, mix_weights, mix_weights, out_weights)
out = MultiHeadAttn(
    n_seq=T,
    n_data=N,
    n_heads=H,
    dim_model=K,
    dim_input=C).process_self(weights, inputs)
print(out.shape)
