from variable_protocols.protocols import Variable
from variable_protocols.variables import var_tensor, dim, bounded_float, ordinal, one_hot, \
    cat_ids, cat_vec


# noinspection PyTypeChecker
# because pycharm sucks
def var_image(h: int, w: int, n_channels: int = 3, int_format: bool = False) -> Variable:
    if int_format:
        return var_tensor(bounded_float(0, 1), {dim("h", h), dim("w", w), dim("channel", n_channels)})
    else:
        return var_tensor(ordinal(256), {dim("h", h), dim("w", w), dim("channel", n_channels)})


# noinspection PyTypeChecker
# because pycharm sucks
def var_sentence_1hot(n_vocab: int) -> Variable:
    return var_tensor(one_hot(n_vocab), {dim("word", None)})


# noinspection PyTypeChecker
# because pycharm sucks
def var_sentence(longest_word_len: int) -> Variable:
    return var_tensor(cat_ids(longest_word_len), {dim("word", None)})


# noinspection PyTypeChecker
# because pycharm sucks
def var_sentence_embeddings(n_vocab: int, size_embedding: int) -> Variable:
    return var_tensor(cat_vec(n_vocab, size_embedding), {dim("word", None)})
