import random
import numpy as np

from src.deep.pipeline.dataset import Dataset
from src.games.Game import Game
from src.games.Policy import MctsPolicy2
from src.games.mcts.Mcts import Mcts
from src.games.state.State import State
from src.power4.P4Board import P4Board
from src.power4.P4Rules import P4Rules


def gen_state(policy=None):
    if not policy:
        policy = MctsPolicy2(n=25)
    history = []
    Game(P4Rules.initial_state(), policy, policy).run(on_state_change=lambda state: history.append(state.copy()))
    return history[-1]  # random.choice(history[-5:])


def state_to_sample(state: State):
    mcts = Mcts(state).run(12)
    evaluation = mcts.tree.root.children()

    x = state.board.to_training()
    y = [0] * P4Board.GRID_WIDTH
    for a in evaluation:
        y[a] = evaluation[a].exploitation_score()
    return x, np.array(y).astype(np.float32)


def sample_generator(n):
    X = []
    Y = []
    for i in range(0, n):
        print('i: ', i)
        state = gen_state()
        x, y = state_to_sample(state)
        print(state)
        X.append(x)
        Y.append(y)
    return X, Y


X, Y = sample_generator(10000)
db = Dataset('p4_dataset_endgames.tfrecords')

db.write(X, Y)
print('done !!')

# Z = db.read()
# print(Z)
