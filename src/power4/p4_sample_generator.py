import random

import numpy as np

from src.deep.MyDataset import MyDataset
from src.games.Game import Game
from src.games.Policy import MctsPolicy2
from src.games.mcts.Mcts import Mcts
from src.games.state.State import State
from src.power4.P4Rules import P4Rules


class SampleGenerator:
    def __init__(self, filename, board_preparation) -> None:
        self.filename = filename
        self.game_generation_policy = MctsPolicy2(n=25)
        self.evaluate = lambda state: Mcts(state).run(100).tree.root.exploitation_score()
        self.board_preparation = board_preparation

    def gen_batch(self, batch_size: int, batch_number: int = 1, target_path='.'):
        for k in range(batch_number):
            samples = self.gen_samples(batch_size)
            name = '{}.{}'.format(self.filename, k)
            db = MyDataset(name, features={'s1', 's2', 'x', 'y'})
            for sample in samples:
                db.add(sample)
            db.remove_dupplicates().save(target_path)
        print('done !')
        return self

    def gen_samples(self, n: int) -> [np.ndarray]:
        samples = []
        for i in range(0, n):
            sample = self._gen_sample()
            print('i: ', i)
            print(sample['s1'])
            print('--[{}]-->'.format(sample['y']))
            print(sample['s2'])
            samples.append(sample)

        return samples

    def _gen_sample(self) -> [np.ndarray]:
        # gen game state
        history = []
        self.game_generation_policy.reset()
        game = Game(P4Rules.initial_state(), self.game_generation_policy, self.game_generation_policy)
        game.run(on_state_change=lambda state: history.append(state.copy()))

        # state selection
        index = random.randint(3, len(history) - 2)
        state1 = history[index]
        state2 = history[index + 1]

        # sample generation
        return {'s1': np.array(state1.board.grid, dtype=np.float32),
                's2': np.array(state2.board.grid, dtype=np.float32),
                'x': (self._gen_sample_input(state1, state2)),
                'y': (self._gen_sample_output(state1, state2))
                }

    def _gen_sample_input(self, state1: State, state2: State) -> np.ndarray:
        p1, p2 = self.board_preparation(state1.board)
        p3, p4 = self.board_preparation(state2.board)
        return np.array([p1, p2, p3, p4], dtype=np.float32)

    def _gen_sample_output(self, state1: State, state2: State):
        evaluation1 = self.evaluate(state1)
        evaluation2 = self.evaluate(state2)
        return np.array([evaluation2 - evaluation1], dtype=np.float32)
