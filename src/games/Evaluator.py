from src.games.state.State import State


class Evaluator:
    def eval(self, state: State) -> [float]:
        raise NotImplementedError()

    def eval_current(self, state: State) -> float:
        return max(self.eval(state))
