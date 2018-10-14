from src.games.state.State import State
from src.games.mcts.Mcts import Mcts


class Policy:
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def play(self, state: State) -> int:
        raise NotImplementedError()

    def update(self, action):
        pass

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


def _do_nothing(mcts: Mcts):
    pass


class MctsPolicy(Policy):

    def __init__(self, n: int, after_play=_do_nothing) -> None:
        super().__init__('mcts_{}'.format(n))
        self.n = n
        self.after_play = after_play

    def play(self, state: State) -> int:
        mcts = Mcts(initial_state=state)
        mcts.run(self.n)
        action = mcts.tree.root.best_action()
        self.after_play(mcts)
        return action


class MctsPolicy2(Policy):

    def __init__(self, n: int, after_play=_do_nothing) -> None:
        super().__init__('mcts_{}'.format(n))
        self.n = n
        self.after_play = after_play
        self.mcts = None

    def update(self, action):
        if self.mcts:
            self.mcts = Mcts(root_node=self.mcts.tree.root.children()[action])

    def play(self, state: State) -> int:
        if self.mcts is None:
            self.mcts = Mcts(initial_state=state)
        assert self.mcts.tree.root.state.board.id() == state.board.id()

        self.mcts.run(self.n)
        action = self.mcts.tree.root.best_action()
        self.after_play(self.mcts)
        return action
