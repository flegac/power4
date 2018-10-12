import random

from src.games.Evaluator import Evaluator
from src.games.state.State import State


def negamax(evaluator: Evaluator, state: State, depth: int, color: int):
    if state is None:
        return -10000000 * color
    if depth == 0 or state.is_terminal:
        return color * evaluator.eval_current(state)

    score = -10000000
    for column_id in state.actions():
        next_state = state.next(column_id)
        score = max(score, -negamax(evaluator, next_state, depth - 1, -color))
    return score


class NegamaxEvaluator(Evaluator):
    def __init__(self, evaluator: Evaluator, depth=3, exploration: float = .25) -> None:
        self.evaluator = evaluator
        self.depth = depth
        self.exploration = exploration

    def eval(self, state: State) -> int:
        actions = state.actions()
        if random.random() < self.exploration:
            action = actions[random.randrange(0, len(actions))]
            state.next(action)
        else:
            choices_dict = {a: negamax(self.evaluator, state.copy().next(a), self.depth, 1) for a in actions}
            action = max(choices_dict, key=choices_dict.get)
            state.next(action)
        return action
