from src.games.state.State import State
from src.games.util.mcts import MCTS


class Policy:

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def play(self, state: State) -> int:
        raise NotImplementedError()

    def name(self):
        return self._name


class Human(Policy):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def play(self, state: State) -> int:
        actions = state.actions()
        player_action = None
        while player_action not in actions:
            try:
                player_action = int(input('Your turn : choose from {}'.format(actions)))
                return player_action
            except:
                print('invalid move !')


class MctsPolicy(Policy):

    def __init__(self, n: int) -> None:
        super().__init__('mcts_{}'.format(n))
        self.n = n

    def play(self, state: State) -> int:
        # print('thinking ...')
        mcts = MCTS(state=state)
        mcts.run(self.n)
        # mcts.stats()
        action = mcts.root.best_move()
        return action
