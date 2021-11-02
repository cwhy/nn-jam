import pickle
from functools import partial

import numpy as np
from bokeh.document import without_document_lock
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure
from einops import repeat, rearrange
from pynng import Pair0
from tornado.ioloop import IOLoop

doc = curdoc()

n_patches_side = 7
w = n_patches_side ** 2
h = n_patches_side ** 2
w2 = 10
h2 = 10
w3 = 28
h3 = 28
if w <= h:
    width_policy = 'fit'
    height_policy = 'max'
else:
    assert w > h
    width_policy = 'max'
    height_policy = 'fit'

eps = 0.00001


def show_img(img, _h, _w):
    _min = img.min()
    _max = img.max()
    img = (img - _min) / (_max - _min)
    img *= 255
    mono_opa = repeat(img, 'h w -> h w c', c=4)
    mono_opa[:, :, :3] = 0
    img = mono_opa
    v_img = np.empty((_h, _w), dtype=np.uint32)
    view = v_img.view(dtype=np.uint8).reshape((_h, _w, 4))
    view[:] = img[:]

    v_img = np.flipud(v_img)
    return [v_img]


def processx(img):
    # img = rearrange(img, '(a b) (c d) -> (a c) (b d)', a=n_patches_side,  c=n_patches_side)
    return show_img(img, h, w)


def processy(img):
    return show_img(img, h2, w2)


def processimg(img):
    print(img.mean())
    return show_img(img, h3, w3)


source = ColumnDataSource(
    data=dict(
        x=processx(np.random.randint(0, 1, (h, w))),
        y=processy(np.random.randint(0, 1, (h2, w2))),
        tr_img=processimg(np.random.randint(0, 1, (h3, w3)))
    ))


def get_figure(_h, _w, name):
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", f"@{name}")],
               tools="",
               toolbar_location=None,
               x_range=(0, _w), y_range=(_h, 0),
               width_policy=width_policy,
               height_policy=height_policy,
               aspect_ratio=_w / _h,
               title=name)
    p.image_rgba(image=name, source=source, x=0, y=_h, dw=_w, dh=_h)
    return p


p1 = get_figure(h, w, "x")
p2 = get_figure(h2, w2, "y")
p3 = get_figure(h3, w3, "tr_img")

p = row([column(p3, p2), p1])


@without_document_lock
def update(new_data):
    source.data.update(dict(
        x=processx(new_data['x']),
        y=processy(new_data['y']),
        tr_img=processimg(new_data['tr_img'])
    ))


doc.add_root(p)

socket = Pair0(listen='tcp://127.0.0.1:54323')


async def loop():
    while True:
        new_data_raw = await socket.arecv()
        new_data = pickle.loads(new_data_raw)
        doc.add_next_tick_callback(partial(update, new_data))


IOLoop.current().spawn_callback(loop)
