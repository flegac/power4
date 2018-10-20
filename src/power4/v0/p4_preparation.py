import os
import random
import numpy as np

from src.deep.pipeline.dataset import Dataset
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
        self.x = []
        self.y = []

    def gen_batch(self, batch_size: int, batch_number: int = 1):
        for k in range(batch_number):
            s1, s2, x, y = self.gen_samples(batch_size)
            assert len(s1) == len(s2) == len(x) == len(y) == batch_size
            db = Dataset().set_data(x, y)
            db.write('{}_n={}.{}.tfrecords'.format(self.filename, batch_size, k))
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

    def concatenate(self, output_name=None):
        if not output_name:
            output_name = self.filename

        dataset = Dataset()
        for file in os.listdir('.'):
            if file.startswith(self.filename):
                print(file)
                dataset.read(file)
        dataset.write('{}_{}.tfrecords'.format(output_name, len(dataset.x)))
        return self

    def _gen_sample(self) -> [np.ndarray]:
        # gen game state
        history = []
        self.game_generation_policy.reset()
        game = Game(P4Rules.initial_state(), self.game_generation_policy, self.game_generation_policy)
        game.run(on_state_change=lambda state: history.append(state.copy()))

        # state selection
        index = random.randrange(-5, -1)
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


def p4_board_preparation(board: P4Board):
    p1_func = np.vectorize(lambda x: 1.0 if (x % 3) == 1 else 0)
    p1 = p1_func(board.grid)

    p2_func = np.vectorize(lambda x: -1.0 if (x % 3) == 2 else 0)
    p2 = p2_func(board.grid)

    # side = 1 if board.current_turn % 2 == 0 else -1
    # initiative_side = np.full((P4Board.GRID_WIDTH, P4Board.GRID_HEIGHT), side)

    return p1, p2


gen = SampleGenerator('p4_testing', board_preparation=p4_board_preparation)

gen.gen_batch(batch_size=2, batch_number=5).concatenate()
