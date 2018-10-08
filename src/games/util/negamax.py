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
