import string
import warnings
from functools import cached_property, lru_cache
from typing import Optional, List, NamedTuple, Tuple, Callable

import jax.numpy as xp
import numpy.typing as npt
from einops import EinopsError, rearrange
from einops.parsing import ParsedExpression
from jax import jit


def _report_axes(axes: set, report_message: str):
    if len(axes) > 0:
        raise EinopsError(report_message.format(axes))


def _product(sequence: List[int]) -> int:
    """ minimalistic product that works both with numbers and symbols. Supports empty lists """
    result = 1
    for element in sequence:
        result *= element
    return result


def mix(pattern: str, weight_shape: str, bias_shape: Optional[str] = None, **axes_lengths):
    left_pattern, right_pattern = pattern.split('->')
    left = ParsedExpression(left_pattern)
    right = ParsedExpression(right_pattern)
    weight = ParsedExpression(weight_shape)
    _report_axes(
        set.difference(right.identifiers, {*left.identifiers, *weight.identifiers}),
        'Unrecognized identifiers on the right side of EinMix {}'
    )

    if left.has_ellipsis or right.has_ellipsis or weight.has_ellipsis:
        raise EinopsError('Ellipsis is not supported in EinMix (right now)')
    if any(x.has_non_unitary_anonymous_axes for x in [left, right, weight]):
        raise EinopsError('Anonymous axes (numbers) are not allowed in EinMix')
    if '(' in weight_shape or ')' in weight_shape:
        raise EinopsError(f'Parenthesis is not allowed in weight shape: {weight_shape}')

    pre_reshape_pattern = None
    pre_reshape_lengths: Tuple[Tuple[str, int], ...] = ()
    post_reshape_pattern = None
    if any(len(group) != 1 for group in left.composition):
        names = []
        for group in left.composition:
            names += group
        composition = ' '.join(names)
        pre_reshape_pattern = f'{left_pattern}->{composition}'
        pre_reshape_lengths = tuple((name, length) for name, length in axes_lengths.items() if name in names)

    if any(len(group) != 1 for group in right.composition):
        names = []
        for group in right.composition:
            names += group
        composition = ' '.join(names)
        post_reshape_pattern = f'{composition}->{right_pattern}'

    for axis in weight.identifiers:
        if axis not in axes_lengths:
            raise EinopsError('Dimension {} of weight should be specified'.format(axis))
    _report_axes(
        set.difference(set(axes_lengths), {*left.identifiers, *weight.identifiers}),
        'Axes {} are not used in pattern',
    )
    _report_axes(
        set.difference(weight.identifiers, {*left.identifiers, *right.identifiers}),
        'Weight axes {} are redundant'
    )
    if len(weight.identifiers) == 0:
        warnings.warn('EinMix: weight has no dimensions (means multiplication by a number)')

    _weight_shape = [axes_lengths[axis] for axis, in weight.composition]
    # single output element is a combination of fan_in input elements
    _fan_in = _product([axes_lengths[axis] for axis, in weight.composition if axis not in right.identifiers])
    if bias_shape is not None:
        if not isinstance(bias_shape, str):
            raise EinopsError('bias shape should be string specifying which axes bias depends on')
        bias = ParsedExpression(bias_shape)
        _report_axes(
            set.difference(bias.identifiers, right.identifiers),
            'Bias axes {} not present in output'
        )
        _report_axes(
            set.difference(bias.identifiers, set(axes_lengths)),
            'Sizes not provided for bias axes {}',
        )

        _bias_shape_list: List[int] = []
        for axes in right.composition:
            for axis in axes:
                if axis in bias.identifiers:
                    _bias_shape_list.append(axes_lengths[axis])
                else:
                    _bias_shape_list.append(1)
        _bias_shape: Optional[Tuple[int, ...]] = tuple(_bias_shape_list)
    else:
        _bias_shape = None
        _bias_input_size = None

    # rewrite einsum expression with single-letter latin identifiers so that
    # expression will be understood by any framework
    mapping2letters_set = {*left.identifiers, *right.identifiers, *weight.identifiers}
    mapping2letters = {k: letter for letter, k in zip(string.ascii_lowercase, mapping2letters_set)}

    def write_flat(axes: list):
        return ''.join(mapping2letters[axis] for axis in axes)

    einsum_pattern: str = '{},{}->{}'.format(
        write_flat(left.flat_axes_order()),
        write_flat(weight.flat_axes_order()),
        write_flat(right.flat_axes_order()),
    )
    return Mix(tuple(_weight_shape),
               _bias_shape,
               einsum_pattern,
               pre_reshape_pattern,
               pre_reshape_lengths,
               post_reshape_pattern)


class MixWeights(NamedTuple):
    w: npt.NDArray
    b: Optional[npt.NDArray]


class Mix(NamedTuple):
    weight_shape: Tuple[int, ...]
    bias_shape: Optional[Tuple[int, ...]]
    einsum_pattern: str
    pre_reshape_pattern: Optional[str]
    pre_reshape_lengths: Tuple[Tuple[str, int], ...]
    post_reshape_pattern: Optional[str]

    @property
    @lru_cache()
    def process(self) -> Callable[[MixWeights, npt.NDArray], npt.NDArray]:
        def _fn(weights: MixWeights, x: npt.NDArray) -> npt.NDArray:
            if self.pre_reshape_pattern is not None:
                params = {k: v for k, v in self.pre_reshape_lengths}
                x = rearrange(x, self.pre_reshape_pattern, **params)
            x = xp.einsum(self.einsum_pattern, x, weights.w)
            if weights.b is not None:
                x += weights.b
            if self.post_reshape_pattern is not None:
                x = rearrange(x, self.post_reshape_pattern)
            return x

        return jit(_fn)
