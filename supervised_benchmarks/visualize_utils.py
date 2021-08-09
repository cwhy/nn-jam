from typing import Optional

import numpy as np
from bokeh.plotting import figure, Figure
from einops import repeat


def view_img_rgba(img: np.ndarray, title: Optional[str]) -> Figure:
    assert len(img.shape) == 3
    assert np.all(img <= 255)
    assert np.all(0 <= img)
    h, w, c = img.shape
    assert c == 4
    v_img = np.empty((h, w), dtype=np.uint32)
    view = v_img.view(dtype=np.uint8).reshape((h, w, 4))
    view[:] = img[:]

    if w < h:
        width_policy = 'fit'
        height_policy = 'max'
    else:
        assert w >= h
        width_policy = 'max'
        height_policy = 'fit'

    v_img = np.flipud(v_img)
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
               tools="",
               toolbar_location=None,
               x_range=(0, w), y_range=(h, 0),
               width_policy=width_policy,
               height_policy=height_policy,
               aspect_ratio=w / h,
               title=title)

    p.image_rgba(image=[v_img], x=0, y=h, dw=w, dh=h)
    return p


def view_2d_mono(img: np.ndarray, label: Optional[str] = None) -> Figure:
    assert len(img.shape) == 2
    h, w = img.shape
    if not 0.2 < h / w < 5:
        raise ValueError("The ratios are too large for 2d plot, consider other type of plots")
    mono_opa = repeat(img, 'h w -> h w c', c=4)
    mono_opa[:, :, :3] = 0
    p = view_img_rgba(mono_opa, label)
    return p
