import numpy as np

from src.games.state.State import State


class Evaluator:

    def __init__(self, w, h, brain_depth) -> None:
        self.weights = np.random.rand(h * brain_depth).reshape((h, brain_depth))
        self.bias = np.random.rand(w * brain_depth).reshape((w, brain_depth))
        self.bias *= .1

    def eval(self, state: State) -> [float]:
        raise NotImplementedError()

    def eval_current(self, state: State) -> float:
        return max(self.eval(state))
