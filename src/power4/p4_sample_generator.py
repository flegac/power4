import random
import numpy as np

from src.deep.dataset import Dataset
from src.games.Game import Game
from src.games.Policy import MctsPolicy2
from src.games.mcts.Mcts import Mcts
from src.games.state.State import State
from src.power4.P4Board import P4Board
from src.power4.P4Rules import P4Rules


class SampleGenerator:
    def __init__(self, filename, board_preparation) -> None:
        self.filename = filename
        self.game_generation_policy = MctsPolicy2(n=25)
        self.evaluate = lambda state: Mcts(state).run(2000).tree.root.exploitation_score()
        self.board_preparation = board_preparation

    def gen_batch(self, batch_size: int, batch_number: int = 1):
        for k in range(batch_number):
            s1, s2, x, y = self.gen_samples(batch_size)
            assert len(s1) == len(s2) == len(x) == len(y) == batch_size
            name = '{}_n={}.{}'.format(self.filename, batch_size, k)
            Dataset(name, x=x, y=y).save()
        print('done !')
        return self

    def gen_samples(self, n: int) -> [np.ndarray]:
        s1 = []
        s2 = []
        x = []
        y = []
        for i in range(0, n):
            _s1, _s2, _x, _y = self._gen_sample()
            s1.append(_s1)
            s2.append(_s2)
            x.append(_x)
            y.append(_y)

            print('i: ', i)
            print(_s1.board)
            print('--[{}]-->'.format(y))
            print(_s2.board)

        return s1, s2, x, y

    def _gen_sample(self) -> [np.ndarray]:
        # gen game state
        history = []
        self.game_generation_policy.reset()
        game = Game(P4Rules.initial_state(), self.game_generation_policy, self.game_generation_policy)
        game.run(on_state_change=lambda state: history.append(state.copy()))

        # state selection
        index = random.randrange(0, -1)
        state1 = history[index]
        state2 = history[index + 1]

        # sample generation
        x = self._gen_sample_input(state1, state2)
        y = self._gen_sample_output(state1, state2)
        return state1, state2, x, y

    def _gen_sample_input(self, state1: State, state2: State) -> np.ndarray:
        p1, p2 = self.board_preparation(state1.board)
        p3, p4 = self.board_preparation(state2.board)
        return np.array([p1, p2, p3, p4], dtype=np.float32)

    def _gen_sample_output(self, state1: State, state2: State):
        evaluation1 = self.evaluate(state1)
        evaluation2 = self.evaluate(state2)
        return np.array([evaluation2 - evaluation1], dtype=np.float32)
