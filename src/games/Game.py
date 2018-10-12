from src.games.state import State
from src.games.Policy import Policy


class Game:
    def __init__(self, state: State, p1: Policy, p2: Policy, turn_callback=None) -> None:
        self.players = [p1, p2]
        self.state = state
        self.history = []
        self.turn_callback = turn_callback

    def run(self):
        current_turn = 0
        while not self.state.is_terminal:
            if self.turn_callback:
                self.turn_callback(self.state)
            current_player = self.players[current_turn % 2]
            action = current_player.play(self.state)
            self.history.append(action)
            self.state.next(action)
            current_turn += 1
        return self
