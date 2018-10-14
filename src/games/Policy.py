from src.games.state.State import State
from src.games.util.Mcts import Mcts


class Policy:

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def play(self, state: State) -> int:
        raise NotImplementedError()

    def name(self):
        return self._name

    def __repr__(self) -> str:
        return self.name()


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


def _show_mcts_stats(mcts: Mcts):
    mcts.stats()


class MctsPolicy(Policy):

    def __init__(self, n: int, after_play=_show_mcts_stats) -> None:
        super().__init__('mcts_{}'.format(n))
        self.n = n
        self.after_play = after_play

    def play(self, state: State) -> int:
        # print('thinking ...')
        mcts = Mcts(initial_state=state)
        mcts.run(self.n)
        action = mcts.root_node.best_action()
        self.after_play(mcts)
        return action
