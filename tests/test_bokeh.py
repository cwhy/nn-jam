import time
from functools import partial
from random import random
from threading import Thread

from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure

# only modify from a Bokeh session callback
source = ColumnDataSource(data=dict(x=[0], y=[0]))


# This is important! Save curdoc() to make sure all threads
# see the same document.

async def update(x, y):
    source.stream(dict(x=[x], y=[y]))


doc = curdoc()
p = figure(x_range=[0, 1], y_range=[0, 1])
l = p.circle(x='x', y='y', source=source)
doc.add_root(p)


def blocking_task():
    i = 0
    while i < 100:
        # do some blocking computation
        time.sleep(0.1)
        x, y = random(), random()
        i += 1

        # but update the document from a callback
        doc.add_next_tick_callback(partial(update, x=x, y=y))


thread = Thread(target=blocking_task)
thread.start()
