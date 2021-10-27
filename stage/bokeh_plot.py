import json

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from functools import partial
from tornado.ioloop import IOLoop
from pynng import Pair0

doc = curdoc()
source = ColumnDataSource(data=dict(x=[], y=[]))
plot = figure(height=500)
plot.line(x='x', y='y', source=source)


def update(new_data):
    source.stream(new_data, rollover=1000)


socket = Pair0(listen='tcp://127.0.0.1:54321')


async def loop():
    while True:
        new_data = await socket.arecv()
        doc.add_next_tick_callback(partial(update, json.loads(new_data.decode("utf-8"))))


doc.add_root(plot)
IOLoop.current().spawn_callback(loop)

