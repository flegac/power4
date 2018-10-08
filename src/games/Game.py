from src.games.state import State
from src.games.Player import Player


class Game:
    def __init__(self, state: State, p1: Player, p2: Player) -> None:
        self.players = [p1, p2]
        self.state = state
        self.history = []

    def run(self):
        current_turn = 0
        while not self.state.is_terminal:
            current_player = self.players[current_turn % 2]
            action = current_player.play(self.state)
            self.history.append(action)
            current_turn += 1
        return self
