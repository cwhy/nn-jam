from variable_protocols.base_variables import Bounded, Ordinal, OneHot, IDs, CategoricalVector
from variable_protocols.labels import L
from variable_protocols.tensorhub import DimFam, Dim, Tensor, V, TensorHub


def var_image(h: int, w: int, n_channels: int = 3, int_format: bool = False) -> TensorHub:
    if int_format:
        return V(Tensor(Bounded(0, 1),
                        frozenset({DimFam(h, L("h")), DimFam(w, L("w")), DimFam(n_channels, L("channel"))})))
    else:
        return V(Ordinal(256) * Dim({"h": h, "w": w, "channel": n_channels}))


def var_sentence_1hot(n_vocab: int) -> TensorHub:
    return V(OneHot(n_vocab) * Dim({"word": None}))


def var_sentence() -> TensorHub:
    return V(IDs("str") * Dim({"word": None}))


def var_sentence_embeddings(n_vocab: int, size_embedding: int) -> TensorHub:
    return V(CategoricalVector(n_vocab, size_embedding) * Dim({"word": None}))
