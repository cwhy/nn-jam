from mcts_python.games.my_connect4 import conn4_env, conn4_net
from mcts_python.games.my_wuziqi import wuziqi_env, wzq_net
from mcts_python.games.my_reversi import reversi_env, reversi_net
from mcts_python.games.my_tictactoe import ttt_env, ttt_net
from mcts_python.games.my_wuziqi_1swap import wuziqi_env as wuziqi_1swap_env, wzq_net as wzq_1swap_net


envs = {
    'TicTacToe': ttt_env,
    'Reversi': reversi_env,
    'WuZiQi': wuziqi_env,
    'WuZiQi_1swap': wuziqi_1swap_env,
    'Connect4': conn4_env,
}

nets = {
    'TicTacToe': ttt_net,
    'Reversi': reversi_net,
    'WuZiQi': wzq_net,
    'WuZiQi_1swap': wzq_1swap_net,
    'Connect4': conn4_net,
}
