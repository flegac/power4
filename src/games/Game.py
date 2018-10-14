from src.games.Policy import Policy
from src.games.state import State


def _do_nothing(state: State):
    pass


class Game:
    def __init__(self, state: State, p1: Policy, p2: Policy, before_play=_do_nothing) -> None:
        self.policies = [p1, p2]
        self.state = state
        self.history = []
        self.before_play = before_play

    def run(self):
        current_turn = 0
        while not self.state.is_terminal:
            self.before_play(self.state)
            current_player = self.policies[current_turn % 2]
            action = current_player.play(self.state)
            self.history.append(action)
            self.state.apply(action)
            current_turn += 1
        return self

    def __repr__(self) -> str:
        p1, p2 = self.policies
        return '{game.state}\n{p1} vs {p2} : {game.state.terminal_result}\n{game.history}'.format(game=self, p1=p1,
                                                                                                  p2=p2)
