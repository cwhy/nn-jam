import numpy as np


def ucb(q: float, c_puct: float, p: float, n: float, na: float):
    return q + c_puct * p * np.sqrt(n) / (1 + na)


def ucb_all(qs: np.array,
            c_puct_normed_by_sum: float,
            ps: np.array,
            n_as: np.array):
    return qs + c_puct_normed_by_sum * ps / (1 + n_as)


# def kl_logits(a, b):
#     kl = ((a - b) * F.softmax(a, dim=-1)).sum() \
#          - torch.logsumexp(a, dim=-1) \
#          + torch.logsumexp(b, dim=-1)
#     return kl
