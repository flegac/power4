import itertools

from src.games.Game import Game
from src.games.state.State import State
from src.games.Player import Player


class Tournament:
    def __init__(self, state: State, game_per_player: int) -> None:
        self.state = state
        self.games = []
        self.game_per_player = game_per_player
        self.players = []

    def set_players(self, *players: [Player]):
        self.players = players

    def run(self):
        for pair in itertools.combinations(self.players, 2):
            p1, p2 = pair
            for i in range(self.game_per_player):
                game = Game(self.state.copy(), p1, p2)
                game.run()
                self.games.append(game.history)
                print(game.state)
                p1, p2 = p2, p1
