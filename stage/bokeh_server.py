import pickle
from functools import partial
import time

import numpy as np
from bokeh.document import without_document_lock
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure
from einops import repeat, rearrange
from pynng import Pair0, Sub0
from tornado.ioloop import IOLoop

doc = curdoc()

n_patches_side = 7
y_dims = 10
pic_shape = (28, 28)
eps = 0.00001
pic_dims = {
    'x': (n_patches_side ** 2, n_patches_side ** 2),
    'y': (y_dims, y_dims),
    'tr_img': pic_shape,
    'tst_img': pic_shape,
}
lines = {
    'loss_tr',
    'epoch'
}
img_processes = {
    k: lambda img, coef=k: show_img(img, coef)
    for k in pic_dims
}

width_policy = 'fit'
height_policy = 'max'

img_processes['x'] = lambda img: show_img(
    rearrange(img, '(a b) (c d) -> (a c) (b d)', a=n_patches_side, c=n_patches_side),
    'x')


def show_img(img, name):
    _h, _w = pic_dims[name]
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


source_pics = ColumnDataSource(
    data={
        k: img_processes[k](np.random.randint(1, 3, v))
        for k, v in pic_dims.items()}
)
source_line = ColumnDataSource(
    data={
        key: [] for key in lines
    }
)


def get_figure(name):
    _h, _w = pic_dims[name]
    _p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", f"@{name}")],
                tools="",
                toolbar_location=None,
                x_range=(0, _w), y_range=(_h, 0),
                width_policy=width_policy,
                height_policy=height_policy,
                aspect_ratio=_w / _h,
                title=name)
    _p.image_rgba(image=name, source=source_pics, x=0, y=_h, dw=_w, dh=_h)
    return _p


def get_lines(name):
    _p = figure(tooltips=[("y", "$y"), ("value", f"@{name}")],
                toolbar_location=None,
                width_policy='max',
                height_policy=height_policy,
                title=name)
    _p.line(x='epoch', y='loss_tr', source=source_line, color="navy", line_width=2)
    #    _p.line(y='tst_loss', source=source_line, color="firebrick", line_width=2)
    return _p


p = row(column(get_figure('tr_img'), get_figure('tst_img')),
        column(row(get_figure('x'), get_figure('y')),
               row(get_lines("losses"))))

time_base = time.time()


@without_document_lock
def update(new_data):
    try:
        source_pics.data.update({k: v(new_data[k]) for k, v in img_processes.items()})
    except Exception as e:
        print(f"some pics not received {e}")

    try:
        lines_data = {k: [v.item()] for k, v in new_data.items() if k in lines}
        lines_data['epoch'] = [time.time() - time_base]
        source_line.stream(lines_data)
    except Exception as e:
        print(f"some lines not received {e}")


doc.add_root(p)

sub = Sub0(listen='tcp://127.0.0.1:54323', recv_timeout=10000)
sub.subscribe(b'')


async def loop():
    while True:
        new_data_raw = await sub.arecv()
        new_data = pickle.loads(new_data_raw)
        doc.add_next_tick_callback(partial(update, new_data))


IOLoop.current().spawn_callback(loop)
