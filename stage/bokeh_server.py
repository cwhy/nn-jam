import pickle
from functools import partial

import numpy as np
from bokeh.document import without_document_lock
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure
from einops import repeat, rearrange
from pynng import Pair0
from tornado.ioloop import IOLoop

doc = curdoc()

w = 16
h = 16
if w <= h:
    width_policy = 'fit'
    height_policy = 'max'
else:
    assert w > h
    width_policy = 'max'
    height_policy = 'fit'


def processimg(img):
    # img = rearrange(img, '(a b) (c d) -> (a c) (b d)', a=4, b=4, c=4, d=4) * 255
    img *= 255
    mono_opa = repeat(img, 'h w -> h w c', c=4)
    mono_opa[:, :, :3] = 0
    img = mono_opa
    v_img = np.empty((h, w), dtype=np.uint32)
    view = v_img.view(dtype=np.uint8).reshape((h, w, 4))
    view[:] = img[:]

    v_img = np.flipud(v_img)
    return [v_img]


source = ColumnDataSource(data=dict(x=processimg(np.random.randint(0, 256, (16, 16)))))

p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
           tools="",
           toolbar_location=None,
           x_range=(0, w), y_range=(h, 0),
           width_policy=width_policy,
           height_policy=height_policy,
           aspect_ratio=w / h,
           title="img")

p.image_rgba(image="x", source=source, x=0, y=h, dw=w, dh=h)


@without_document_lock
def update(new_data):
    source.data.update(dict(x=processimg(new_data['x'])))


doc.add_root(p)

socket = Pair0(listen='tcp://127.0.0.1:54322')


async def loop():
    while True:
        new_data_raw = await socket.arecv()
        new_data = pickle.loads(new_data_raw)
        doc.add_next_tick_callback(partial(update, new_data))


IOLoop.current().spawn_callback(loop)
