from src.games.state.State import State
from src.games.util.mcts import MCTS


class Evaluator:
    def eval(self, state: State) -> [float]:
        raise NotImplementedError()

    def eval_current(self, state: State) -> float:
        return max(self.eval(state))


class MctsEvaluator:
    def __init__(self, n) -> None:
        self.n = n

    def eval(self, state: State) -> [float]:
        mcts = MCTS(state=state)
        mcts.run(self.n)
