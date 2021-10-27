import pickle
import time

import numpy as np
from pynng import Pair0

from stage.protocol import Stage


with Pair0(dial='tcp://127.0.0.1:54322') as socket:
    stage = Stage(socket)
    for i in range(1000):
        time.sleep(1)
        stuff = np.random.randint(0, 256, (20, 20))
        stage.socket.send(pickle.dumps(dict(x=stuff)))
