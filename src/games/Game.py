from src.games.Policy import Policy
from src.games.state import State


def _do_nothing(state: State):
    pass


class Game:
    def __init__(self, state: State, p1: Policy, p2: Policy) -> None:
        self.policies = [p1, p2]
        self.state = state
        self.history = []

    def run(self, on_state_change=_do_nothing):
        current_turn = 0
        while not self.state.is_terminal:
            on_state_change(self.state)
            current_player = self.policies[current_turn % 2]
            action = current_player.play(self.state)
            for p in set(self.policies):
                p.update(action)
            self.history.append(action)
            self.state.apply(action)
            current_turn += 1
        on_state_change(self.state)
        return self

    def __repr__(self) -> str:
        p1, p2 = self.policies
        return '{game.state}\n{p1} vs {p2} : {game.state.terminal_result}\n{game.history}'.format(game=self, p1=p1,
                                                                                                  p2=p2)
