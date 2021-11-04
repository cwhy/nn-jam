from multiprocessing import Process
from multiprocessing import Pipe
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from mcts_python.protocols import Action
from mcts_python.games.gridboard_utils import StateBoard


class WebAgent:
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w
        a, b = Pipe()
        self.server = a
        self.agent = b
        self.proc = None
        self.state_ = None

    def get_actor(self, ag_id: int):
        self.run_(ag_id)

        def _get_actions(s: StateBoard, render: bool = False) -> Action:
            grid = s.get_array.reshape(self.h, self.w)
            grid = np.where(grid == -1, 100, grid)
            self.agent.send(grid.ravel().tolist())
            print("game waiting")
            y, x = self.agent.recv()
            return y * self.w + x

        return _get_actions

    def run_(self, player):
        app = FastAPI()
        app.mount("/dist", StaticFiles(directory="./dist/"), name="public")

        @app.get("/")
        async def read_index():
            return RedirectResponse("/dist/index.html")

        @app.get("/board_init")
        async def board_init():
            return {'player': player,
                    'w': self.w,
                    'h': self.h}

        @app.get("/board_update")
        async def board_update():
            if self.state_ is None:
                state = self.server.recv()
                self.state_ = state
            return {'board_state': self.state_,
                    'message': "Your turn"}

        class PiecePut(BaseModel):
            x: int
            y: int
            player: int

        @app.post("/piece_put")
        async def assign_state(pp: PiecePut):
            print(pp)
            # state[pp.y, pp.x] = pp.player
            self.server.send((pp.y, pp.x))
            # print("server waiting")
            state = self.server.recv()
            self.state_ = state
            return {'board_state': state,
                    'message': "Your turn"}

        self.proc = Process(
            target=lambda: uvicorn.run(app,
                                       host="0.0.0.0",
                                       port=8000 + player),
            daemon=True)
        self.proc.start()
