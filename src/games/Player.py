import random

from src.games.Evaluator import Evaluator
from src.games.state.State import State
from src.games.util.negamax import negamax


class Player:
    def __init__(self, evaluator: Evaluator, depth=3, exploration: float = .25) -> None:
        self.evaluator = evaluator
        self.depth = depth
        self.exploration = exploration

    def play(self, state: State) -> int:
        actions = state.actions()
        if random.random() < self.exploration:
            action = actions[random.randrange(0, len(actions))]
            state.next(action)
        else:
            choices_dict = {a: negamax(self.evaluator, state.copy().next(a), self.depth, 1) for a in actions}
            action = max(choices_dict, key=choices_dict.get)
            state.next(action)
        return action
